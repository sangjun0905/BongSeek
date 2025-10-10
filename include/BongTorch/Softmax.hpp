#pragma once

#include "Core.hpp" 
#include <cmath>
#include <limits>
#include <memory>
#include <vector>
#include "../NumBong/Tensor.hpp" 

namespace bs {
class Softmax : public Function { // bs::Function에서 Function으로 수정
private:
    int axis_;
public:
    // Softmax Function의 생성자: Softmax를 계산할 축(axis)을 받습니다.
    explicit Softmax(int axis = -1) : axis_(axis) {}

    // Function::forward 오버라이딩 (TensorData 타입으로 통일)
    std::vector<Tensor> forward(const std::vector<Tensor>& xs) override {
        const Tensor& x = xs[0];
        const auto shape = x.getShape();
        const std::size_t rank = shape.size();
        const int axis = axis_ < 0 ? static_cast<int>(rank) + axis_ : axis_;
        if (rank != 3 || axis != 2) {
            throw std::invalid_argument("Softmax: only supports rank-3 tensors along the last axis");
        }

        Tensor y(shape);
        const std::size_t batch = shape[0];
        const std::size_t seq = shape[1];
        const std::size_t depth = shape[2];

        for (std::size_t b = 0; b < batch; ++b) {
            for (std::size_t s = 0; s < seq; ++s) {
                float max_val = -std::numeric_limits<float>::infinity();
                for (std::size_t d = 0; d < depth; ++d) {
                    const float val = static_cast<float>(x(b, s, d));
                    if (val > max_val) {
                        max_val = val;
                    }
                }

                float sum_exp = 0.0f;
                std::vector<float> exp_buffer(depth);
                for (std::size_t d = 0; d < depth; ++d) {
                    const float shifted = static_cast<float>(x(b, s, d)) - max_val;
                    const float e = std::exp(shifted);
                    exp_buffer[d] = e;
                    sum_exp += e;
                }

                if (sum_exp == 0.0f) {
                    const float value = 1.0f / static_cast<float>(depth);
                    for (std::size_t d = 0; d < depth; ++d) {
                        y(b, s, d) = TensorValueType(value);
                    }
                } else {
                    const float inv_sum = 1.0f / sum_exp;
                    for (std::size_t d = 0; d < depth; ++d) {
                        y(b, s, d) = TensorValueType(exp_buffer[d] * inv_sum);
                    }
                }
            }
        }

        return { y };
    }
};

// Softmax Function을 쉽게 사용할 수 있도록 헬퍼 함수를 정의합니다.
inline std::shared_ptr<Variable> softmax(const std::shared_ptr<Variable>& x, int axis = -1) {
    auto f = std::make_shared<Softmax>(axis);
    // (*f) 오버로딩을 통해 Function 호출
    auto outs = (*f)(std::vector<std::shared_ptr<Variable>>{x});
    return outs; 
}

} // namespace bs
