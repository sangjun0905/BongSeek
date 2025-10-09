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
};

} // namespace bs
