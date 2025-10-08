#pragma once 

#include "Core.hpp"

class RoPE : public bs::Function {
public:
    std::vector<nb::Array> forward(const std::vector<nb::Array>& xs) override {
        const nb::Array& x = xs[0]; //
        const nb::Array& C = xs[1]; // Cos 
        const nb::Array& S = xs[2]; // Sin 

        int d_k = x.shape()[x.ndim() - 1];
        int d_half = d_k / 2;
        nb::Array x_A = nb::split(x, 0, d_half); // 앞쪽 절반 (x_0)
        nb::Array x_B = nb::split(x, 1, d_half); // 뒤쪽 절반 (x_1)

        nb::Array term1 = bs::mul(x_A, C) - bs::mul(x_B, S);

        nb::Array term2 = bs::mul(x_B, C) + bs::mul(x_A, S);

        nb::Array y = nb::concat({ term1, term2 }, x.ndim() - 1);

        return { y };
    }

    std::vector<std::shared_ptr<bs::Variable>> backward(const std::vector<std::shared_ptr<bs::Variable>>& gys) override {
        return { nullptr, nullptr, nullptr };
    }
};

// Function Wrapper
inline std::shared_ptr<bs::Variable> rope(const std::shared_ptr<bs::Variable>& x,
    const std::shared_ptr<bs::Variable>& C,
    const std::shared_ptr<bs::Variable>& S) {
    auto f = std::make_shared<RoPE>();
    auto outs = (*f)(std::vector<std::shared_ptr<bs::Variable>>{x, C, S});
    return outs[0];
}