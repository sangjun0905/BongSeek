#pragma once

#include "Core.hpp"          
#include "Module.hpp" 
#include "MatMul.hpp" // MatMul Function과 matmul() 래퍼 함수가 여기에 있다고 가정
#include <memory>

namespace bs { // 모든 모듈과 Function을 네임스페이스로 묶는 것이 좋습니다.

class Linear : public Module { // Module 상속
private:
    std::shared_ptr<Parameter> W; // 가중치 행렬
    std::shared_ptr<Parameter> b; // 편향 벡터
    bool use_bias;

public:
    Linear(int in_features, int out_features, bool bias = true) 
        : use_bias(bias) 
    {
        // 1. 가중치 W
        TensorShape w_shape = {static_cast<std::size_t>(out_features), 
                               static_cast<std::size_t>(in_features), 
                               1}; 
        // nb::create_rand_tensor 대신, Shape에 맞는 TensorData 객체만 생성합니다.
        W = Parameter::create(TensorData(w_shape), "weight"); 
        register_parameter("weight", W); 

        if (use_bias) {
            // 2. 편향 b
            TensorShape b_shape = {1, 1, static_cast<std::size_t>(out_features)}; 
            // nb::create_zeros_tensor 대신, Shape에 맞는 TensorData 객체만 생성합니다.
            b = Parameter::create(TensorData(b_shape), "bias");
            register_parameter("bias", b); 
        } else {
            b = nullptr;
        }
    }

    // Module::forward 인터페이스 구현
    std::shared_ptr<Variable> forward(const std::shared_ptr<Variable>& x) override {
        // Linear 연산: y = x @ W^T + b
        
        // 1. W Transpose (W: [Out, In, 1] -> W_T: [In, Out, 1] 형태로 마지막 두 축 전치)
        // NOTE: TensorData의 transpose()가 마지막 두 축을 전치한다고 가정
        auto W_t = Variable::create(W->data.transpose()); 
        
        // 2. 행렬 곱: x @ W^T
        auto output = matmul(x, W_t);
        
        // 3. 편향 덧셈 (Variable의 + 연산자 오버로딩 사용)
        if (use_bias) {
            // 편향 b는 [1, 1, Out] 형태이므로 브로드캐스팅으로 덧셈이 수행됩니다.
            output = output + b; 
        }
        
        return output;
    }
};

} // namespace bs