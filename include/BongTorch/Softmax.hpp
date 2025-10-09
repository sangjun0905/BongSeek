#pragma once

#include "Core.hpp" 
#include <memory>
<<<<<<< HEAD
#include "../NumBong/Tensor.hpp" 

namespace bs {
class Softmax : public Function { // bs::Function에서 Function으로 수정
=======
#include <vector>
#include <cmath> // For std::exp
#include <limits> // For std::numeric_limits

namespace bs {

class Softmax : public Function {
>>>>>>> origin/BongTorchJW
private:
    int axis_;
public:
    explicit Softmax(int axis = -1) : axis_(axis) {}

<<<<<<< HEAD
    // Function::forward 오버라이딩 (TensorData 타입으로 통일)
    std::vector<Tensor> forward(const std::vector<Tensor>& xs) override {
        const Tensor& x = xs[0]; // 입력 텐서 (TensorData로 통일)

        // 1. 안정화 단계 (Stability): x에서 최대값을 빼줍니다.
        // nb::max(tensor, axis, keep_dims=true)를 가정합니다.
        Tensor x_max = nb::max(x, axis_, true); 
        Tensor x_shifted = x - x_max; 

        // 2. 분자 계산: exp(x_shifted)
        Tensor numerator = nb::exp(x_shifted); 
        
        // 3. 분모 계산: sum(numerator)
        // nb::sum(tensor, axis, keep_dims=true)를 가정합니다.
        Tensor denominator = nb::sum(numerator, axis_, true); 
        
        // 4. 최종 계산: numerator / denominator
        Tensor y = numerator / denominator;
=======
    std::vector<Tensor> forward(const std::vector<Tensor>& xs) override {
        const Tensor& x = xs[0];
        auto x_shape = x.getShape();
        Tensor output(x_shape);

        if (axis_ != -1 && axis_ != static_cast<int>(x.ndim() - 1)) {
            throw std::runtime_error("Softmax only supports last axis (-1) for now.");
        }

        auto B = x_shape[0];
        auto S = x_shape[1];
        auto D = x_shape[2];
>>>>>>> origin/BongTorchJW

        for (size_t b = 0; b < B; ++b) {
            for (size_t s = 0; s < S; ++s) {
                // 1. Find max for stability
                float max_val = -std::numeric_limits<float>::infinity();
                for (size_t d = 0; d < D; ++d) {
                    max_val = std::max(max_val, static_cast<float>(x(b, s, d)));
                }

                // 2. Calculate exp and sum of exps
                float sum_exp = 0.0f;
                std::vector<float> exps(D);
                for (size_t d = 0; d < D; ++d) {
                    exps[d] = std::exp(static_cast<float>(x(b, s, d)) - max_val);
                    sum_exp += exps[d];
                }

                // 3. Divide by sum
                for (size_t d = 0; d < D; ++d) {
                    output(b, s, d) = nb::BFloat16(exps[d] / sum_exp);
                }
            }
        }

        return { output };
    }
};

inline std::shared_ptr<Variable> softmax(const std::shared_ptr<Variable>& x, int axis = -1) {
    auto f = std::make_shared<Softmax>(axis);
<<<<<<< HEAD
    // (*f) 오버로딩을 통해 Function 호출
    auto outs = (*f)(std::vector<std::shared_ptr<Variable>>{x});
    return std::make_shared<Variable>(outs[0]); 
=======
    return (*f)({x});
>>>>>>> origin/BongTorchJW
}

} // namespace bs
