#pragma once

#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

#include "Module.hpp"
#include "Core.hpp"
#include "Linear.hpp"
#include "Softmax.hpp"

namespace bs {

class GQAAttention : public Module {
private:
    std::size_t num_heads_;
    std::size_t num_kv_heads_;
    std::size_t head_dim_;
    std::size_t kv_repeats_;
    nb::BFloat16 eps_{nb::BFloat16(1e-6f)};

    std::shared_ptr<Linear> WQ_;
    std::shared_ptr<Linear> WK_;
    std::shared_ptr<Linear> WV_;
    std::shared_ptr<Linear> WO_;

    std::shared_ptr<Parameter> q_norm_weight_;
    std::shared_ptr<Parameter> k_norm_weight_;

    Tensor make_gamma() const {
        TensorShape gamma_shape = {1, 1, head_dim_};
        Tensor gamma(gamma_shape);
        gamma.fill(TensorValueType(1.0f));
        return gamma;
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

    Tensor rms_norm(const Tensor& src,
                        const Tensor& gamma,
                        std::size_t local_head_dim) const {
        Tensor out(src.getShape());
        const std::size_t batch = src.getShape()[0];
        const std::size_t seq = src.getShape()[1];
        const std::size_t dim = src.getShape()[2];
        const nb::BFloat16 dim_b(dim);
        const nb::BFloat16 one(1.0f);

        for (std::size_t b = 0; b < batch; ++b) {
            for (std::size_t s = 0; s < seq; ++s) {
                nb::BFloat16 sum_sq(0.0f);
                for (std::size_t d = 0; d < dim; ++d) {
                    const nb::BFloat16 v = src(b, s, d);
                    sum_sq += v * v;
                }
                const nb::BFloat16 mean_sq = sum_sq / dim_b;
                const nb::BFloat16 denom = nb::bfloat16_sqrt(mean_sq + eps_);
                const nb::BFloat16 inv_rms = one / denom;
                for (std::size_t d = 0; d < dim; ++d) {
                    const std::size_t gamma_idx = d % local_head_dim;
                    out(b, s, d) = src(b, s, d) * inv_rms * gamma(0, 0, gamma_idx);
                }
            }
        }

        return out;
    }

    Tensor compute_scores(const Tensor& q,
                              const Tensor& k,
                              std::size_t batch,
                              std::size_t seq) const {
        const std::size_t total_heads = batch * num_heads_;
        Tensor scores({total_heads, seq, seq});
        const nb::BFloat16 head_dim_b(head_dim_);
        const nb::BFloat16 scale = nb::BFloat16(1.0f) / nb::bfloat16_sqrt(head_dim_b);

        for (std::size_t bh = 0; bh < total_heads; ++bh) {
            for (std::size_t i = 0; i < seq; ++i) {
                for (std::size_t j = 0; j < seq; ++j) {
                    nb::BFloat16 dot(0.0f);
                    for (std::size_t d = 0; d < head_dim_; ++d) {
                        dot += q(bh, i, d) * k(bh, j, d);
                    }
                    scores(bh, i, j) = dot * scale;
                }
            }
        }

        return scores;
    }

    Tensor reshape_back(const Tensor& src,
                            std::size_t batch,
                            std::size_t seq) const {
        Tensor out({batch, seq, num_heads_ * head_dim_});
        for (std::size_t b = 0; b < batch; ++b) {
            for (std::size_t s = 0; s < seq; ++s) {
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

    Tensor apply_attention(const Tensor& scores,
                               const Tensor& values,
                               std::size_t batch,
                               std::size_t seq) const {
        const std::size_t total_heads = batch * num_heads_;
        Tensor out({total_heads, seq, head_dim_});

        for (std::size_t bh = 0; bh < total_heads; ++bh) {
            for (std::size_t i = 0; i < seq; ++i) {
                for (std::size_t d = 0; d < head_dim_; ++d) {
                    nb::BFloat16 acc(0.0f);
                    for (std::size_t j = 0; j < seq; ++j) {
                        acc += scores(bh, i, j) * values(bh, j, d);
                    }
                    out(bh, i, d) = acc;
                }
            }
        }

        return out;
    }

public:
    GQAAttention() {};

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

        add_module("WQ", WQ_);
        add_module("WK", WK_);
        add_module("WV", WV_);
        add_module("WO", WO_);

        q_norm_weight_ = Parameter::create(make_gamma(), "q_layernorm.weight");
        k_norm_weight_ = Parameter::create(make_gamma(), "k_layernorm.weight");
        register_parameter("q_layernorm.weight", q_norm_weight_);
        register_parameter("k_layernorm.weight", k_norm_weight_);
    }

    std::shared_ptr<Variable> forward(const std::shared_ptr<Variable>& x) override {
        const auto shape = x->shape();
        const std::size_t batch = shape[0];
        const std::size_t seq = shape[1];

        auto safe_linear = [](const std::shared_ptr<Linear>& linear,
                               const std::shared_ptr<Variable>& input,
                               const char* label) {
            try {
                return (*linear)(input);
            } catch (const std::exception& e) {
                throw std::runtime_error(std::string(label) + " failed: " + e.what());
            }
        };

        auto q_var = safe_linear(WQ_, x, "GQAAttention.WQ");
        auto k_var = safe_linear(WK_, x, "GQAAttention.WK");
        auto v_var = safe_linear(WV_, x, "GQAAttention.WV");

        Tensor q_norm = rms_norm(q_var->data, q_norm_weight_->data, head_dim_);
        Tensor k_norm = rms_norm(k_var->data, k_norm_weight_->data, head_dim_);
        Tensor q_heads = reshape_to_heads(q_norm, batch, seq, num_heads_, head_dim_);
        Tensor k_heads_base = reshape_to_heads(k_norm, batch, seq, num_kv_heads_, head_dim_);
        Tensor v_heads_base = reshape_to_heads(v_var->data, batch, seq, num_kv_heads_, head_dim_);
        Tensor k_heads = repeat_kv_heads(k_heads_base, batch, seq);
        Tensor v_heads = repeat_kv_heads(v_heads_base, batch, seq);

        Tensor scores = compute_scores(q_heads, k_heads, batch, seq);
        auto scores_var = Variable::create(scores, "scores");
        scores_var = softmax(scores_var, 2);

        Tensor context_heads = apply_attention(scores_var->data, v_heads, batch, seq);
        Tensor context = reshape_back(context_heads, batch, seq);

        auto out_var = Variable::create(context, "attention_output");
        return safe_linear(WO_, out_var, "GQAAttention.WO");
    }

    void loadWeights(std::istream& file, const MetadataMap& metadata){
        MetadataMap WQ_meta;
        MetadataMap WK_meta;
        MetadataMap WV_meta;
        MetadataMap WO_meta;
        MetadataMap k_layernorm_meta; // normalization 구현 시 weight추가
        MetadataMap q_layernorm_meta; // normalization 구현 시 weight추가

        for(auto& [key, value] : metadata) {
            if(key.compare(0, 7, "q_proj.") == 0) {
                WQ_meta[key.substr(7)] = value; 
            } 
            else if (key.compare(0, 7, "k_proj.") == 0) {
                WK_meta[key.substr(7)] = value; 
            } 
            else if (key.compare(0, 7, "v_proj.") == 0) {
                WV_meta[key.substr(7)] = value; 
            } 
            else if (key.compare(0, 9, "out_proj.") == 0) {
                WO_meta[key.substr(9)] = value; 
            } 
            else if (key.compare(0, 12, "k_layernorm.") == 0) {
                k_layernorm_meta[key.substr(12)] = value; // normalization 구현 시 weight추가
            }
            else if (key.compare(0, 12, "q_layernorm.") == 0) {
                q_layernorm_meta[key.substr(12)] = value; // normalization 구현 시 weight추가
            }
            // normalization 구현 시 weight추가
            
        }

        WQ_->loadWeights(file, WQ_meta);
        WK_->loadWeights(file, WK_meta);
        WV_->loadWeights(file, WV_meta);
        WO_->loadWeights(file, WO_meta);
        if (auto it = q_layernorm_meta.find("weight"); it != q_layernorm_meta.end()) {
            const std::string label = "GQAAttention." + q_norm_weight_->name;
            load_tensor_data_checked(label, q_norm_weight_->data, file, it->second);
        } else {
            std::cerr << "[GQAAttention] q_layernorm.weight 메타데이터가 없어 로딩을 건너뜁니다.\n";
        }
        if (auto it = k_layernorm_meta.find("weight"); it != k_layernorm_meta.end()) {
            const std::string label = "GQAAttention." + k_norm_weight_->name;
            load_tensor_data_checked(label, k_norm_weight_->data, file, it->second);
        } else {
            std::cerr << "[GQAAttention] k_layernorm.weight 메타데이터가 없어 로딩을 건너뜁니다.\n";
        }
        // normalization 구현 시 weight추가
    }
};

} // namespace bs
