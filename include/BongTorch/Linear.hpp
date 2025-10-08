#pragma once
#include "Core.hpp"          
//#include "matmul_function.hpp" 
#include <memory>

namespace bs {

class Linear : public Module { // Module 상속 (W, b 상태 관리)
public:
    std::shared_ptr<Parameter> W; // 가중치 (Parameter)
    std::shared_ptr<Parameter> b; // 편향 (Parameter)

    Linear(int in_features, int out_features) {
        // W 초기화 (랜덤 초기화)
        TensorShape w_shape = {1, static_cast<std::size_t>(in_features), static_cast<std::size_t>(out_features)};
        W = Parameter::create(nb::create_rand_tensor(w_shape)); 
        
        // b 초기화 (0 초기화)
        TensorShape b_shape = {1, 1, static_cast<std::size_t>(out_features)};
        b = Parameter::create(nb::create_zeros_tensor(b_shape)); 
    }

    // forward
    std::shared_ptr<Variable> forward(const std::shared_ptr<Variable>& x) override {

        auto y_matmul = matmul(x, W); // xW
        auto y = y_matmul + b; // xW + b

        return y;
    }
};

} // namespace bs