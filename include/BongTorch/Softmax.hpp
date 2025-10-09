#pragma once

#include "Core.hpp" 
#include <memory>
// nb::max, nb::exp, nb::sum 함수가 정의된 NumBong.hpp 파일이 필요합니다.
// #include "NumBong.hpp" 

namespace bs { // 💡 네임스페이스 bs 추가

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
    
    // NOTE: backward는 학습 시 필수적이지만, 현재 추론 전용을 가정하고 생략합니다.
};

// Softmax Function을 쉽게 사용할 수 있도록 헬퍼 함수를 정의합니다.
inline std::shared_ptr<Variable> softmax(const std::shared_ptr<Variable>& x, int axis = -1) {
    auto f = std::make_shared<Softmax>(axis);
    // (*f) 오버로딩을 통해 Function 호출
    auto outs = (*f)(std::vector<std::shared_ptr<Variable>>{x});
    return outs[0]; 
}

} // namespace bs