#pragma once

#include "core.hpp" 

class Softmax : public Function {
private:
    // TensorData는 nb::Tensor<float, 3>의 타입 별칭입니다.
    using TensorData = typename Variable::TensorData; 
    
    int axis_;
public:
    // Softmax Function의 생성자: Softmax를 계산할 축(axis)을 받습니다. 기본값은 -1 (마지막 축).
    explicit Softmax(int axis = -1) : axis_(axis) {}

    // Function 클래스의 forward 메서드를 오버라이딩합니다.
    std::vector<TensorData> forward(const std::vector<TensorData>& xs) override {
        const TensorData& x = xs[0];

        // 1. 안정화 단계 (Stability): x에서 최대값을 빼줍니다.
        // 이는 exp() 계산 시 숫자가 너무 커져 오버플로우가 발생하는 것을 방지합니다.
        // nb::max(tensor, axis, keep_dims=true)를 가정합니다.
        TensorData x_max = nb::max(x, axis_, true); 
        TensorData x_shifted = x - x_max; // 브로드캐스팅을 통해 x의 각 요소에서 최대값(x_max)을 뺌

        // 2. 분자 계산: exp(x_shifted)
        TensorData numerator = nb::exp(x_shifted); 
        
        // 3. 분모 계산: sum(numerator)
        // nb::sum(tensor, axis, keep_dims=true)를 가정합니다.
        TensorData denominator = nb::sum(numerator, axis_, true); 
        
        // 4. 최종 계산: numerator / denominator
        TensorData y = numerator / denominator;

        return { y };
    }
    // 추론 전용이므로 backward는 구현하지 않습니다.
};

// Softmax Function을 쉽게 사용할 수 있도록 헬퍼 함수를 정의합니다.
inline std::shared_ptr<Variable> softmax(const std::shared_ptr<Variable>& x, int axis = -1) {
    auto f = std::make_shared<Softmax>(axis);
    // Function::operator()를 호출하고 결과를 가져옵니다.
    auto outs = (*f)({x});
    return outs[0]; 
}