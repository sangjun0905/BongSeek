#pragma once 

#include "Core.hpp"
#include <memory>
#include "../NumBong/Tensor.hpp" 

namespace bs {

class SiLU : public Function { // bs::Function에서 Function으로 수정
public:
    // forward: SiLU(x) = x * sigma(x) = x / (1 + exp(-x))
    // xs[0] = x (입력 텐서)
    std::vector<TensorData> forward(const std::vector<TensorData>& xs) override {
        const TensorData& x = xs[0]; //

        // 1. exp(-x)
        TensorData exp_neg_x = nb::exp(-x); // nb::exp 함수가 NumBong.hpp에 정의되어야 함

        // 2. 분모 계산: 1 + exp(-x)
        TensorData denominator = 1.0 + exp_neg_x;

        // 3. 최종 결과: x / denominator
        TensorData y = x / denominator;

        return { y };
    }
};

// Function Wrapper (bs::silu)
inline std::shared_ptr<Variable> silu(const std::shared_ptr<Variable>& x) {
    auto f = std::make_shared<SiLU>();
    auto outs = (*f)(std::vector<std::shared_ptr<Variable>>{x});
    return outs[0];
}

} // namespace bs