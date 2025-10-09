#pragma once

#include "Core.hpp"
#include "MatMul.hpp"
#include <memory>

namespace bs {

class Linear : public Module {
private:
    std::shared_ptr<Parameter> W;
    std::shared_ptr<Parameter> b;
    bool use_bias;

public:
    Linear(int in_features, int out_features, bool bias = true) 
        : use_bias(bias) 
    {
        TensorShape w_shape = {1,
                               static_cast<std::size_t>(out_features), 
                               static_cast<std::size_t>(in_features)}; 
        W = Parameter::create(Tensor(w_shape), "weight"); 
        register_parameter("weight", W); 

        if (use_bias) {
            TensorShape b_shape = {1, 1, static_cast<std::size_t>(out_features)}; 
            b = Parameter::create(Tensor(b_shape), "bias");
            register_parameter("bias", b); 
        } else {
            b = nullptr;
        }
    }

    std::shared_ptr<Variable> forward(const std::shared_ptr<Variable>& x) override {
        // W shape is (1, out, in). We need (1, in, out) for matmul.
        auto W_t = Variable::create(W->data.transpose(1, 2)); 
        auto output = matmul(x, W_t);
        
        if (use_bias) {
            // Broadcasting add is not supported yet.
        }
        
        return output;
    }

    std::shared_ptr<Parameter> weight() const { return W; }
    std::shared_ptr<Parameter> bias() const { return b; }

    void loadWeights(std::istream& file, const MetadataMap& metadata)
    {
        auto weight_it = metadata.find("weight");
        if (weight_it == metadata.end()) {
            std::cerr << "[Linear] weight 메타데이터를 찾을 수 없습니다.\n";
            return;
        }

        const auto& weight_info = weight_it->second;
        W->data.loadWeight(file,
                           static_cast<std::streamoff>(weight_info.offset_start),
                           static_cast<std::streamoff>(weight_info.offset_end));

        if (use_bias && b) {
            auto bias_it = metadata.find("bias");
            if (bias_it != metadata.end()) {
                const auto& bias_info = bias_it->second;
                b->data.loadWeight(file,
                                   static_cast<std::streamoff>(bias_info.offset_start),
                                   static_cast<std::streamoff>(bias_info.offset_end));
            } else {
                std::cerr << "[Linear] bias 메타데이터가 없어 bias 로딩을 건너뜁니다.\n";
            }
        }
    }
};

} // namespace bs
