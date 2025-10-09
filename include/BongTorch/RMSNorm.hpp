#pragma once

#include "Core.hpp"
#include "Module.hpp"
#include "RMSNormFunction.hpp"

namespace bs {

class RMSNorm : public Module {
    string name;
public:
    std::shared_ptr<Parameter> weight;
    RMSNorm() {};

    RMSNorm(const string& prefix, int dim) {
        name = prefix;
        weight = Parameter::create(nb::ones({(size_t)dim}));
    }

    std::shared_ptr<Variable> forward(const std::shared_ptr<Variable>& x) override {
        return rms_norm(x, weight);
    }
};

} // namespace bs
