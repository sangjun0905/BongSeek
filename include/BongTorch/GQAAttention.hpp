#pragma once

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

#include "Core.hpp"
#include "Linear.hpp"
#include "RMSNorm.hpp"

namespace bs {

class GQAAttention : public Module {
private:
    using BF16 = nb::BFloat16;

    std::size_t num_heads_;
    std::size_t num_kv_heads_;
    std::size_t head_dim_;
    std::size_t kv_repeats_;

    std::shared_ptr<Linear> WQ_;
    std::shared_ptr<Linear> WK_;
    std::shared_ptr<Linear> WV_;
    std::shared_ptr<Linear> WO_;

    std::shared_ptr<RMSNorm> q_norm_;
    std::shared_ptr<RMSNorm> k_norm_;

    bool cache_enabled_ = false;
    bool cache_initialized_ = false;
    Tensor k_cache_;
    Tensor v_cache_;

    static BF16 value_at(const Tensor& tensor,
                         std::size_t i0,
                         std::size_t i1,
                         std::size_t i2) {
        return tensor(i0, i1, i2);
    }

    static Tensor reshape_to_heads(const Tensor& src,
                                   std::size_t batch,
                                   std::size_t seq,
                                   std::size_t heads,
                                   std::size_t head_dim) {
        Tensor out({batch * heads, seq, head_dim});
        for (std::size_t b = 0; b < batch; ++b) {
            for (std::size_t s = 0; s < seq; ++s) {
                for (std::size_t h = 0; h < heads; ++h) {
                    const std::size_t base = h * head_dim;
                    const std::size_t dst_b = b * heads + h;
                    for (std::size_t d = 0; d < head_dim; ++d) {
                        out(dst_b, s, d) = src(b, s, base + d);
                    }
                }
            }
        }
        return out;
    }

    Tensor repeat_kv_heads(const Tensor& src,
                           std::size_t batch,
                           std::size_t seq) const {
        Tensor out({batch * num_heads_, seq, head_dim_});
        for (std::size_t b = 0; b < batch; ++b) {
            for (std::size_t kv = 0; kv < num_kv_heads_; ++kv) {
                for (std::size_t rep = 0; rep < kv_repeats_; ++rep) {
                    const std::size_t h = kv * kv_repeats_ + rep;
                    const std::size_t src_b = b * num_kv_heads_ + kv;
                    const std::size_t dst_b = b * num_heads_ + h;
                    for (std::size_t s = 0; s < seq; ++s) {
                        for (std::size_t d = 0; d < head_dim_; ++d) {
                            out(dst_b, s, d) = src(src_b, s, d);
                        }
                    }
                }
            }
        }
        return out;
    }

    Tensor append_seq(const Tensor& base,
                      const Tensor& addition) const {
        if (base.size() == 0) {
            return addition;
        }

        const auto base_shape = base.getShape();
        const auto add_shape = addition.getShape();

        if (base_shape[0] != add_shape[0] || base_shape[2] != add_shape[2]) {
            throw std::invalid_argument("GQAAttention: incompatible cache shapes");
        }

        Tensor out({base_shape[0], base_shape[1] + add_shape[1], base_shape[2]});

        for (std::size_t b = 0; b < base_shape[0]; ++b) {
            for (std::size_t s = 0; s < base_shape[1]; ++s) {
                for (std::size_t d = 0; d < base_shape[2]; ++d) {
                    out(b, s, d) = base(b, s, d);
                }
            }
        }

        for (std::size_t b = 0; b < add_shape[0]; ++b) {
            for (std::size_t s = 0; s < add_shape[1]; ++s) {
                for (std::size_t d = 0; d < add_shape[2]; ++d) {
                    out(b, base_shape[1] + s, d) = addition(b, s, d);
                }
            }
        }

        return out;
    }

    Tensor compute_scores(const Tensor& q,
                          const Tensor& k,
                          std::size_t batch,
                          std::size_t q_seq,
                          std::size_t k_seq) const {
        const std::size_t total_heads = batch * num_heads_;
        Tensor scores({total_heads, q_seq, k_seq});
        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));

        for (std::size_t bh = 0; bh < total_heads; ++bh) {
            for (std::size_t i = 0; i < q_seq; ++i) {
                for (std::size_t j = 0; j < k_seq; ++j) {
                    float dot = 0.0f;
                    for (std::size_t d = 0; d < head_dim_; ++d) {
                        const float qv = static_cast<float>(value_at(q, bh, i, d));
                        const float kv = static_cast<float>(value_at(k, bh, j, d));
                        dot += qv * kv;
                    }
                    scores(bh, i, j) = static_cast<TensorValueType>(BF16(dot * scale));
                }
            }
        }

        return scores;
    }

    Tensor softmax_last_dim(const Tensor& src) const {
        Tensor out(src.getShape());
        const auto shape = src.getShape();
        const std::size_t outer0 = shape[0];
        const std::size_t outer1 = shape[1];
        const std::size_t dim    = shape[2];

        for (std::size_t i0 = 0; i0 < outer0; ++i0) {
            for (std::size_t i1 = 0; i1 < outer1; ++i1) {
                float max_val = static_cast<float>(value_at(src, i0, i1, 0));
                for (std::size_t d = 1; d < dim; ++d) {
                    max_val = std::max(max_val,
                                        static_cast<float>(value_at(src, i0, i1, d)));
                }
                float sum = 0.0f;
                for (std::size_t d = 0; d < dim; ++d) {
                    float shifted = static_cast<float>(value_at(src, i0, i1, d)) - max_val;
                    float ex = std::exp(shifted);
                    out(i0, i1, d) = static_cast<TensorValueType>(BF16(ex));
                    sum += ex;
                }
                const float inv_sum = (sum > 0.0f) ? 1.0f / sum : 0.0f;
                for (std::size_t d = 0; d < dim; ++d) {
                    float v = static_cast<float>(out(i0, i1, d));
                    out(i0, i1, d) = static_cast<TensorValueType>(BF16(v * inv_sum));
                }
            }
        }

        return out;
    }

    Tensor apply_attention(const Tensor& scores,
                           const Tensor& values,
                           std::size_t batch,
                           std::size_t q_seq,
                           std::size_t k_seq) const {
        const std::size_t total_heads = batch * num_heads_;
        Tensor out({total_heads, q_seq, head_dim_});

        for (std::size_t bh = 0; bh < total_heads; ++bh) {
            for (std::size_t i = 0; i < q_seq; ++i) {
                for (std::size_t d = 0; d < head_dim_; ++d) {
                    float acc = 0.0f;
                    for (std::size_t j = 0; j < k_seq; ++j) {
                        const float sv = static_cast<float>(value_at(scores, bh, i, j));
                        const float vv = static_cast<float>(value_at(values, bh, j, d));
                        acc += sv * vv;
                    }
                    out(bh, i, d) = static_cast<TensorValueType>(BF16(acc));
                }
            }
        }

        return out;
    }

    Tensor reshape_back(const Tensor& src,
                        std::size_t batch,
                        std::size_t q_seq) const {
        Tensor out({batch, q_seq, num_heads_ * head_dim_});
        for (std::size_t b = 0; b < batch; ++b) {
            for (std::size_t s = 0; s < q_seq; ++s) {
                for (std::size_t h = 0; h < num_heads_; ++h) {
                    const std::size_t src_b = b * num_heads_ + h;
                    const std::size_t base = h * head_dim_;
                    for (std::size_t d = 0; d < head_dim_; ++d) {
                        out(b, s, base + d) = src(src_b, s, d);
                    }
                }
            }
        }
        return out;
    }

public:
    GQAAttention(int input_dim,
                 int num_heads,
                 int num_kv_heads,
                 int head_dim)
        : num_heads_(static_cast<std::size_t>(num_heads)),
          num_kv_heads_(static_cast<std::size_t>(num_kv_heads)),
          head_dim_(static_cast<std::size_t>(head_dim)) {

        if (num_heads_ == 0 || num_kv_heads_ == 0) {
            throw std::invalid_argument("GQAAttention: head counts must be positive");
        }
        if (num_heads_ % num_kv_heads_ != 0) {
            throw std::invalid_argument("GQAAttention: num_heads must be divisible by num_kv_heads");
        }

        kv_repeats_ = num_heads_ / num_kv_heads_;

        const int q_dim = static_cast<int>(num_heads_ * head_dim_);
        const int kv_dim = static_cast<int>(num_kv_heads_ * head_dim_);

        WQ_ = std::make_shared<Linear>(input_dim, q_dim, false);
        WK_ = std::make_shared<Linear>(input_dim, kv_dim, false);
        WV_ = std::make_shared<Linear>(input_dim, kv_dim, false);
        WO_ = std::make_shared<Linear>(q_dim, input_dim, false);

        q_norm_ = std::make_shared<RMSNorm>(q_dim);
        k_norm_ = std::make_shared<RMSNorm>(kv_dim);

        add_module("WQ", WQ_);
        add_module("WK", WK_);
        add_module("WV", WV_);
        add_module("WO", WO_);
        add_module("q_norm", q_norm_);
        add_module("k_norm", k_norm_);
    }

    void enable_kv_cache(bool enable = true) {
        cache_enabled_ = enable;
        if (!enable) {
            reset_kv_cache();
        }
    }

    void reset_kv_cache() {
        cache_initialized_ = false;
        k_cache_ = Tensor();
        v_cache_ = Tensor();
    }

    std::shared_ptr<Variable> forward(const std::shared_ptr<Variable>& x) override {
        const auto shape = x->shape();
        const std::size_t batch = shape[0];
        const std::size_t seq = shape[1];

        auto q_proj = (*WQ_)(x);
        auto q_var = (*q_norm_)(q_proj);

        auto k_proj = (*WK_)(x);
        auto k_var = (*k_norm_)(k_proj);

        auto v_var = (*WV_)(x);

        Tensor q_heads = reshape_to_heads(q_var->data, batch, seq, num_heads_, head_dim_);
        Tensor k_heads_base = reshape_to_heads(k_var->data, batch, seq, num_kv_heads_, head_dim_);
        Tensor v_heads_base = reshape_to_heads(v_var->data, batch, seq, num_kv_heads_, head_dim_);

        Tensor k_heads = repeat_kv_heads(k_heads_base, batch, seq);
        Tensor v_heads = repeat_kv_heads(v_heads_base, batch, seq);

        if (cache_enabled_) {
            if (!cache_initialized_) {
                k_cache_ = k_heads;
                v_cache_ = v_heads;
                cache_initialized_ = true;
            } else {
                k_cache_ = append_seq(k_cache_, k_heads);
                v_cache_ = append_seq(v_cache_, v_heads);
            }
            k_heads = k_cache_;
            v_heads = v_cache_;
        } else {
            reset_kv_cache();
        }

        const std::size_t q_seq = q_heads.getShape()[1];
        const std::size_t k_seq = k_heads.getShape()[1];

        Tensor scores = compute_scores(q_heads, k_heads, batch, q_seq, k_seq);
        Tensor scores_soft = softmax_last_dim(scores);
        Tensor context_heads = apply_attention(scores_soft, v_heads, batch, q_seq, k_seq);
        Tensor context = reshape_back(context_heads, batch, q_seq);

        return (*WO_)(Variable::create(context, "attention_output"));
    }
};

} // namespace bs
