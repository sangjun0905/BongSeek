#pragma once 

#include "Core.hpp"
#include <memory>

// 💡 추가 필요 헤더: nb::exp 함수가 정의된 헤더를 포함해야 합니다.
// #include "NumBong.hpp" 

namespace bs { // 💡 네임스페이스 bs 추가

class SiLU : public Function { // bs::Function에서 Function으로 수정
public:
    // forward: SiLU(x) = x * sigma(x) = x / (1 + exp(-x))
    // xs[0] = x (입력 텐서)
    std::vector<TensorData> forward(const std::vector<TensorData>& xs) override {
        const TensorData& x = xs[0]; // 💡 nb::Array 대신 TensorData로 통일

        // 1. exp(-x)
        TensorData exp_neg_x = nb::exp(-x); // nb::exp 함수가 NumBong.hpp에 정의되어야 함

        // 2. 분모 계산: 1 + exp(-x)
        TensorData denominator = 1.0 + exp_neg_x;

        // 3. 최종 결과: x / denominator
        TensorData y = x / denominator;

        return { y };
    }

    // backward (추론 전용이므로 nullptr 유지. 학습 시에는 실제 기울기 계산 필요)
    std::vector<std::shared_ptr<Variable>> backward(const std::vector<std::shared_ptr<Variable>>& gys) override {
        // dL/dx 기울기 계산은 학습 시 필요합니다.
        // 현재는 추론 전용을 가정하여 nullptr을 반환합니다.
        return { nullptr };
    }
};

// Function Wrapper (bs::silu)
inline std::shared_ptr<Variable> silu(const std::shared_ptr<Variable>& x) {
    auto f = std::make_shared<SiLU>();
    auto outs = (*f)(std::vector<std::shared_ptr<Variable>>{x});
    return outs[0];
}

} // namespace bs