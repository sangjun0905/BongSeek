#pragma once
#include "core.hpp" 
#include "../NumBong/Tensor.hpp"

namespace bs {

class MatMul : public Function {
public:
    std::vector<TensorData> forward(const std::vector<TensorData>& xs) override {
               TensorData Y = xs[0].matmul(xs[1]); 
        return { Y };
    }
};

// Function Wrapper
inline std::shared_ptr<Variable> matmul(const std::shared_ptr<Variable>& x, const std::shared_ptr<Variable>& w) {
    auto f = std::make_shared<MatMul>();
    auto outs = (*f)(std::vector<std::shared_ptr<Variable>>{x, w}); 
    return outs[0];
}

} // namespace bs