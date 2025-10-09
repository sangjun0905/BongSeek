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

    void loadWeights(std::istream& file, const MetadataMap& metadata)
    {
        long long startoffset = metadata.at("weight").offset_start;
        long long endoffset = metadata.at("weight").offset_end;
        weight->data.loadWeight(file, startoffset, endoffset);
    }
};

} // namespace bs
