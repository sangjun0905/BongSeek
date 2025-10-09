#pragma once

#include "Core.hpp" 
#include <memory>
#include "../NumBong/Tensor.hpp" 

namespace bs {
class Softmax : public Function { // bs::Function에서 Function으로 수정
private:
    int axis_;
public:
    // Softmax Function의 생성자: Softmax를 계산할 축(axis)을 받습니다.
    explicit Softmax(int axis = -1) : axis_(axis) {}

    // Function::forward 오버라이딩 (TensorData 타입으로 통일)
    std::vector<TensorData> forward(const std::vector<TensorData>& xs) override {
        const TensorData& x = xs[0]; // 입력 텐서 (TensorData로 통일)

        // 1. 안정화 단계 (Stability): x에서 최대값을 빼줍니다.
        // nb::max(tensor, axis, keep_dims=true)를 가정합니다.
        TensorData x_max = nb::max(x, axis_, true); 
        TensorData x_shifted = x - x_max; 

        // 2. 분자 계산: exp(x_shifted)
        TensorData numerator = nb::exp(x_shifted); 
        
        // 3. 분모 계산: sum(numerator)
        // nb::sum(tensor, axis, keep_dims=true)를 가정합니다.
        TensorData denominator = nb::sum(numerator, axis_, true); 
        
        // 4. 최종 계산: numerator / denominator
        TensorData y = numerator / denominator;

        return { y };
    }
};

// Softmax Function을 쉽게 사용할 수 있도록 헬퍼 함수를 정의합니다.
inline std::shared_ptr<Variable> softmax(const std::shared_ptr<Variable>& x, int axis = -1) {
    auto f = std::make_shared<Softmax>(axis);
    // (*f) 오버로딩을 통해 Function 호출
    auto outs = (*f)(std::vector<std::shared_ptr<Variable>>{x});
    return outs[0]; 
}

} // namespace bs