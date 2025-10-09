#pragma once 

#include "Core.hpp"
#include <memory>
#include "../NumBong/Tensor.hpp" 

namespace bs {

class SiLU : public Function {
public:
    std::vector<Tensor> forward(const std::vector<Tensor>& xs) override {
        const Tensor& x = xs[0];

        // 1. Manual negation of x
        Tensor neg_x(x.getShape());
        const auto* x_data = x.data();
        auto* neg_x_data = neg_x.data();
        for (size_t i = 0; i < x.size(); ++i) {
            neg_x_data[i] = nb::BFloat16(-static_cast<float>(x_data[i]));
        }

        // 2. exp(-x)
        Tensor exp_neg_x = nb::exp(neg_x);

        // 3. Denominator: 1 + exp(-x)
        Tensor denominator = 1.0 + exp_neg_x;

        // 4. Final result: x / denominator
        Tensor y = x / denominator;

        return { y };
    }
};

// Function Wrapper (bs::silu)
inline std::shared_ptr<Variable> silu(const std::shared_ptr<Variable>& x) {
    auto f = std::make_shared<SiLU>();
    return (*f)({x});
}

} // namespace bs
