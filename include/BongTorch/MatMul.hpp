#pragma once 

#include "Core.hpp" 
#include <memory>

class MatMul : public Function {
public:
    // Function::forward 오버라이딩
    std::vector<TensorData> forward(const std::vector<TensorData>& xs) override {
        // nb::Tensor::matmul을 사용하여 행렬 곱셈을 수행합니다.
        // xs[0] @ xs[1]
        return { xs[0].matmul(xs[1]) };
    }

    // Function::backward 오버라이딩 (자동 미분 지원)
    std::vector<std::shared_ptr<Variable>> backward(const std::vector<std::shared_ptr<Variable>>& gys) override {
        auto A = inputs[0]; // A (xs[0])
        auto B = inputs[1]; // B (xs[1])
        auto dL_dY = gys[0]; // dL/dY (상위 계층 기울기)

        // 1. dL/dA 계산: dL/dY @ B^T
        // B^T: B->data.transpose()는 마지막 두 축을 전치한다고 가정
        auto B_T = Variable::create(B->data.transpose()); 
        
        // dL/dA = matmul(dL_dY, B_T)
        // dL/dA를 계산하기 위해 matmul 래퍼 함수를 재귀적으로 사용합니다.
        // NOTE: matmul 래퍼 함수가 정의되어야 합니다.
        auto dL_dA = matmul(dL_dY, B_T); 

        // 2. dL/dB 계산: A^T @ dL/dY
        // A^T: A->data.transpose()는 마지막 두 축을 전치한다고 가정
        auto A_T = Variable::create(A->data.transpose()); 

        // dL/dB = matmul(A_T, dL_dY)
        auto dL_dB = matmul(A_T, dL_dY);
        
        // 브로드캐스팅 처리 (선택적): 만약 배치 차원이 브로드캐스팅되었다면, sum_to 처리가 필요합니다.
        // 현재는 배치 차원이 일치한다고 가정하고 (예: B, S, Dk) 그대로 반환합니다.
        
        return { dL_dA, dL_dB };
    }
};

// Function Wrapper: 사용자 편의를 위한 래퍼 함수
inline std::shared_ptr<Variable> matmul(const std::shared_ptr<Variable>& a, const std::shared_ptr<Variable>& b) {
    auto f = std::make_shared<MatMul>();
    // Function::operator()를 호출하고 결과를 가져옵니다.
    auto outs = (*f)(std::vector<std::shared_ptr<Variable>>{a, b});
    return outs[0]; // 단일 출력을 반환
}