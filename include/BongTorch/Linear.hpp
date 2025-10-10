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
        const std::string weight_label = "Linear." + W->name;
        load_tensor_data_checked(weight_label, W->data, file, weight_info);

        if (use_bias && b) {
            auto bias_it = metadata.find("bias");
            if (bias_it != metadata.end()) {
                const auto& bias_info = bias_it->second;
                const std::string bias_label = "Linear." + b->name;
                load_tensor_data_checked(bias_label, b->data, file, bias_info);
            } else {
                std::cerr << "[Linear] bias 메타데이터가 없어 bias 로딩을 건너뜁니다.\n";
            }
        }
    }
};

} // namespace bs
