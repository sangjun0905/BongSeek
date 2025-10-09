#pragma once

#include "Core.hpp"
#include <cmath> // For std::sqrt

namespace bs {

class RMSNorm : public Module {
private:
    std::shared_ptr<Parameter> weight;
    double epsilon_ = 1e-5;
    int dim_;

public:
    RMSNorm(int dim) : dim_(dim) {
        TensorShape weight_shape = {1, 1, static_cast<size_t>(dim)};
        auto weight_tensor = Tensor(weight_shape);
        weight_tensor.fill(nb::BFloat16(1.0f)); // Initialize with ones
        weight = Parameter::create(weight_tensor, "weight");
        register_parameter("weight", weight);
    }

    std::shared_ptr<Variable> forward(const std::shared_ptr<Variable>& x_var) override {
        const auto& x = x_var->data;
        auto x_shape = x.getShape();
        auto B = x_shape[0];
        auto S = x_shape[1];
        auto D = x_shape[2];

        if (D != static_cast<size_t>(dim_)) {
            throw std::runtime_error("RMSNorm: input dimension mismatch.");
        }

        Tensor output(x_shape);

        for (size_t b = 0; b < B; ++b) {
            for (size_t s = 0; s < S; ++s) {
                // 1. Calculate mean of squares for the feature vector
                float sum_sq = 0.0f;
                for (size_t d = 0; d < D; ++d) {
                    float val = static_cast<float>(x(b, s, d));
                    sum_sq += val * val;
                }
                float mean_sq = sum_sq / D;

                // 2. Calculate rsqrt(mean_sq + epsilon)
                float rrms = 1.0f / std::sqrt(mean_sq + epsilon_);

                // 3. Normalize and apply weight
                for (size_t d = 0; d < D; ++d) {
                    float normalized_val = static_cast<float>(x(b, s, d)) * rrms;
                    float weighted_val = normalized_val * static_cast<float>(weight->data(0, 0, d));
                    output(b, s, d) = nb::BFloat16(weighted_val);
                }
            }
        }

        return Variable::create(output, "rms_norm_output");
    }
};

} // namespace bs