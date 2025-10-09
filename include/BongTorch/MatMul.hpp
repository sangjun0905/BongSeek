#pragma once 

#include "Core.hpp" 
#include <memory>

namespace bs {
    
class MatMul : public Function {
public:
    // Function::forward 오버라이딩
    std::vector<TensorData> forward(const std::vector<TensorData>& xs) override {
        // nb::Tensor::matmul을 사용하여 행렬 곱셈을 수행합니다.
        // xs[0] @ xs[1]
        return { xs[0].matmul(xs[1]) };
    }
};

// Function Wrapper: 사용자 편의를 위한 래퍼 함수
inline std::shared_ptr<Variable> matmul(const std::shared_ptr<Variable>& a, const std::shared_ptr<Variable>& b) {
    auto f = std::make_shared<MatMul>();
    // Function::operator()를 호출하고 결과를 가져옵니다.
    auto outs = (*f)(std::vector<std::shared_ptr<Variable>>{a, b});
    return outs[0]; // 단일 출력을 반환
}

}