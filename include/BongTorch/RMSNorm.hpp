#pragma once

#include "Core.hpp"
#include "Module.hpp"
#include "RMSNormFunction.hpp"

namespace bs {

class RMSNorm : public Module {
public:
    std::shared_ptr<Parameter> weight;

    RMSNorm(int dim) {
        weight = Parameter::create(nb::ones({(size_t)dim}));
    }

    std::shared_ptr<Variable> forward(const std::shared_ptr<Variable>& x) override {
        return rms_norm(x, weight);
    }
};

} // namespace bs
