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
    float eps_ = 1e-6f;

    std::shared_ptr<Linear> WQ_;
    std::shared_ptr<Linear> WK_;
    std::shared_ptr<Linear> WV_;
    std::shared_ptr<Linear> WO_;

    std::shared_ptr<Parameter> q_norm_weight_;
    std::shared_ptr<Parameter> k_norm_weight_;

    TensorData make_gamma() const {
        TensorShape gamma_shape = {1, 1, head_dim_};
        TensorData gamma(gamma_shape);
        gamma.fill(static_cast<TensorValueType>(1.0f));
        return gamma;
    }

    static TensorData reshape_to_heads(const TensorData& src,
                                       std::size_t batch,
                                       std::size_t seq,
                                       std::size_t heads,
                                       std::size_t head_dim) {
        TensorData out({batch * heads, seq, head_dim});
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

    TensorData repeat_kv_heads(const TensorData& src,
                               std::size_t batch,
                               std::size_t seq) const {
        TensorData out({batch * num_heads_, seq, head_dim_});
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

    TensorData rms_norm(const TensorData& src,
                        const TensorData& gamma,
                        std::size_t local_head_dim) const {
        TensorData out(src.getShape());
        const std::size_t batch = src.getShape()[0];
        const std::size_t seq = src.getShape()[1];
        const std::size_t dim = src.getShape()[2];

        for (std::size_t b = 0; b < batch; ++b) {
            for (std::size_t s = 0; s < seq; ++s) {
                float sum_sq = 0.0f;
                for (std::size_t d = 0; d < dim; ++d) {
                    const float v = src(b, s, d);
                    sum_sq += v * v;
                }
                const float denom = std::sqrt(sum_sq / static_cast<float>(dim) + eps_);
                const float inv_rms = 1.0f / denom;
                for (std::size_t d = 0; d < dim; ++d) {
                    const std::size_t gamma_idx = d % local_head_dim;
                    out(b, s, d) = src(b, s, d) * inv_rms * gamma(0, 0, gamma_idx);
                }
            }
        }

        return out;
    }

    TensorData compute_scores(const TensorData& q,
                              const TensorData& k,
                              std::size_t batch,
                              std::size_t seq) const {
        const std::size_t total_heads = batch * num_heads_;
        TensorData scores({total_heads, seq, seq});
        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));

        for (std::size_t bh = 0; bh < total_heads; ++bh) {
            for (std::size_t i = 0; i < seq; ++i) {
                for (std::size_t j = 0; j < seq; ++j) {
                    float dot = 0.0f;
                    for (std::size_t d = 0; d < head_dim_; ++d) {
                        dot += q(bh, i, d) * k(bh, j, d);
                    }
                    scores(bh, i, j) = dot * scale;
                }
            }
        }

        return scores;
    }

    TensorData reshape_back(const TensorData& src,
                            std::size_t batch,
                            std::size_t seq) const {
        TensorData out({batch, seq, num_heads_ * head_dim_});
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

    TensorData apply_attention(const TensorData& scores,
                               const TensorData& values,
                               std::size_t batch,
                               std::size_t seq) const {
        const std::size_t total_heads = batch * num_heads_;
        TensorData out({total_heads, seq, head_dim_});

        for (std::size_t bh = 0; bh < total_heads; ++bh) {
            for (std::size_t i = 0; i < seq; ++i) {
                for (std::size_t d = 0; d < head_dim_; ++d) {
                    float acc = 0.0f;
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

        auto q_var = (*WQ_)(x);
        auto k_var = (*WK_)(x);
        auto v_var = (*WV_)(x);

        TensorData q_norm = rms_norm(q_var->data, q_norm_weight_->data, head_dim_);
        TensorData k_norm = rms_norm(k_var->data, k_norm_weight_->data, head_dim_);

        TensorData q_heads = reshape_to_heads(q_norm, batch, seq, num_heads_, head_dim_);
        TensorData k_heads_base = reshape_to_heads(k_norm, batch, seq, num_kv_heads_, head_dim_);
        TensorData v_heads_base = reshape_to_heads(v_var->data, batch, seq, num_kv_heads_, head_dim_);

        TensorData k_heads = repeat_kv_heads(k_heads_base, batch, seq);
        TensorData v_heads = repeat_kv_heads(v_heads_base, batch, seq);

        TensorData scores = compute_scores(q_heads, k_heads, batch, seq);
        auto scores_var = Variable::create(scores, "scores");
        scores_var = softmax(scores_var, 2);

        TensorData context_heads = apply_attention(scores_var->data, v_heads, batch, seq);
        TensorData context = reshape_back(context_heads, batch, seq);

        return (*WO_)(Variable::create(context, "attention_output"));
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
            else if (key.compare(0, 15, "k_layernorm.") == 0) {
                k_layernorm_meta[key.substr(15)] = value; // normalization 구현 시 weight추가
            }
            else if (key.compare(0, 15, "q_layernorm.") == 0) {
                q_layernorm_meta[key.substr(15)] = value; // normalization 구현 시 weight추가
            }
            // normalization 구현 시 weight추가
            
        }

        //WQ_->loadWeights(file, WQ_meta);
        //WK_->loadWeights(file, WK_meta);
        //WV_->loadWeights(file, WV_meta);
        //WO_->loadWeights(file, WO_meta);
        // normalization 구현 시 weight추가
    }
};

} // namespace bs
