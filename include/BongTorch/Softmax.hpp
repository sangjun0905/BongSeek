#pragma once

#include "Core.hpp"

class Softmax : public bs::Function {
private:
    int axis_;
public:
    explicit Softmax(int axis = -1) : axis_(axis) {}

    // forward
    std::vector<nb::Array> forward(const std::vector<nb::Array>& xs) override {
        const nb::Array& x = xs[0];

        nb::Array x_max = nb::max(x, axis_, true);
        nb::Array x_shifted = x - x_max; // 최댓값 빼주기

        nb::Array numerator = nb::exp(x_shifted);
        nb::Array denominator = nb::sum(numerator, axis_, true);
        nb::Array y = numerator / denominator;

        return { y };
    };
};

inline std::shared_ptr<Variable> softmax(const std::shared_ptr<Variable>& x, int axis = -1) {
    auto f = std::make_shared<Softmax>(axis);
    auto outs = (*f)(std::vector<std::shared_ptr<Variable>>{x});
    return outs[0];
}