#pragma once 

#include "Core.hpp"

class SiLU : public bs::Function {
public:
    // forward
    std::vector<nb::Array> forward(const std::vector<nb::Array>& xs) override {
        const nb::Array& x = xs[0]; // 입력 텐서

        // e^(-x)
        nb::Array exp_neg_x = nb::exp(-x);

        // 3. 분모 계산: 1 + e^(-x) (스칼라 덧셈은 nb::Array의 오버로딩을 가정)
        // 이 결과는 시그모이드 함수 sigma(x)의 분모입니다.
        nb::Array denominator = 1.0 + exp_neg_x;

        // 4. 최종 결과: x * sigma(x) = x / (1 + e^(-x))
        // 오버로딩된 나누기 연산을 사용합니다.
        nb::Array y = x / denominator;

        return { y };
    }

    // backward
    std::vector<std::shared_ptr<bs::Variable>> backward(const std::vector<std::shared_ptr<bs::Variable>>& gys) override {
        // 추론 전용이므로 기울기 계산 로직은 실행되지 않음
        return { nullptr };
    }
};

// Function Wrapper (bs::silu)
inline std::shared_ptr<bs::Variable> silu(const std::shared_ptr<bs::Variable>& x) {
    auto f = std::make_shared<SiLU>();
    auto outs = (*f)(std::vector<std::shared_ptr<bs::Variable>>{x});
    return outs[0];
}