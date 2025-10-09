#pragma once

#include "Core.hpp"
#include "Module.hpp"
#include "RMSNormFunction.hpp"

namespace bs {

class RMSNorm : public Module {
public:
    std::shared_ptr<Parameter> weight;

    RMSNorm(int dim) {
        Tensor<BFloat16, 1> a((size_t) dim);
        weight = Parameter::create(a);
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
