#pragma once

#include "Core.hpp"     

class RMSNormFunction : public bs::Function {
private:
    const double epsilon_ = 1e-5; // epsilon

public:
    // Forward: y = x * rsqrt(mean(x^2) + eps) * gamma
    std::vector<nb::Array> forward(const std::vector<nb::Array>& xs) override {
        const nb::Array& x = xs[0]; // x
        const nb::Array& gamma = xs[1]; // gamma

        nb::Array x1 = x ^ 2.0;
        nb::Array mean_x1 = nb::mean(x1);
        nb::Array rrms = nb::rsqrt(mean_x1 + epsilon);
        nb::Array y = x * rrms * gamma;

        return { y };
    }
};

inline std::shared_ptr<Variable> rms_norm(const std::shared_ptr<Variable>& x, const std::shared_ptr<Variable>& g) {
    auto f = std::make_shared<RMSNormFunction>();
    auto outs = (*f)(std::vector<std::shared_ptr<Variable>>{x, g});
    return outs[0];
}
