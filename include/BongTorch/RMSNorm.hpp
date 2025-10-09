#pragma once

#include <cmath>

#include "Core.hpp"
#include "Module.hpp"

namespace bs {

class RMSNorm : public Module {
private:
    std::shared_ptr<Parameter> weight_;
    std::size_t dim_;
    float eps_;

public:
    explicit RMSNorm(std::size_t dim, float eps = 1e-6f)
        : dim_(dim), eps_(eps) {
        Tensor gamma({1, 1, dim_});
        for (std::size_t d = 0; d < dim_; ++d) {
            gamma(0, 0, d) = static_cast<TensorValueType>(nb::BFloat16(1.0f));
        }
        weight_ = Parameter::create(gamma, "weight");
        register_parameter("weight", weight_);
    }

    std::shared_ptr<Variable> forward(const std::shared_ptr<Variable>& x) override {
        const auto& x_data = x->data;
        const auto shape = x_data.getShape();
        if (shape[2] != dim_) {
            throw std::invalid_argument("RMSNorm: input last dimension mismatch");
        }

        Tensor out(shape);
        const auto& gamma = weight_->data;

        const std::size_t batch = shape[0];
        const std::size_t seq = shape[1];

        for (std::size_t b = 0; b < batch; ++b) {
            for (std::size_t s = 0; s < seq; ++s) {
                float sum_sq = 0.0f;
                for (std::size_t d = 0; d < dim_; ++d) {
                    const float v = static_cast<float>(x_data(b, s, d));
                    sum_sq += v * v;
                }
                const float denom = std::sqrt(sum_sq / static_cast<float>(dim_) + eps_);
                const float inv_rms = (denom > 0.0f) ? 1.0f / denom : 0.0f;
                for (std::size_t d = 0; d < dim_; ++d) {
                    const float gamma_v = static_cast<float>(gamma(0, 0, d));
                    const float x_v = static_cast<float>(x_data(b, s, d));
                    out(b, s, d) = static_cast<TensorValueType>(nb::BFloat16(x_v * inv_rms * gamma_v));
                }
            }
        }
        return Variable::create(out);
    }

    std::shared_ptr<Parameter> weight() const { return weight_; }
};

} // namespace bs
