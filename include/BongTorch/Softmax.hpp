#pragma once

#include "Core.hpp" 
#include <memory>
#include <vector>
#include <cmath> // For std::exp
#include <limits> // For std::numeric_limits

namespace bs {

class Softmax : public Function {
private:
    int axis_;
public:
    explicit Softmax(int axis = -1) : axis_(axis) {}

    std::vector<Tensor> forward(const std::vector<Tensor>& xs) override {
        const Tensor& x = xs[0];
        auto x_shape = x.getShape();
        Tensor output(x_shape);

        if (axis_ != -1 && axis_ != static_cast<int>(x.ndim() - 1)) {
            throw std::runtime_error("Softmax only supports last axis (-1) for now.");
        }

        auto B = x_shape[0];
        auto S = x_shape[1];
        auto D = x_shape[2];

        for (size_t b = 0; b < B; ++b) {
            for (size_t s = 0; s < S; ++s) {
                // 1. Find max for stability
                float max_val = -std::numeric_limits<float>::infinity();
                for (size_t d = 0; d < D; ++d) {
                    max_val = std::max(max_val, static_cast<float>(x(b, s, d)));
                }

                // 2. Calculate exp and sum of exps
                float sum_exp = 0.0f;
                std::vector<float> exps(D);
                for (size_t d = 0; d < D; ++d) {
                    exps[d] = std::exp(static_cast<float>(x(b, s, d)) - max_val);
                    sum_exp += exps[d];
                }

                // 3. Divide by sum
                for (size_t d = 0; d < D; ++d) {
                    output(b, s, d) = nb::BFloat16(exps[d] / sum_exp);
                }
            }
        }

        return { output };
    }
};

inline std::shared_ptr<Variable> softmax(const std::shared_ptr<Variable>& x, int axis = -1) {
    auto f = std::make_shared<Softmax>(axis);
    return (*f)({x});
}

} // namespace bs
