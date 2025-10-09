#pragma once 

#include "Core.hpp"
#include <memory>
#include <stdexcept>

namespace bs {

class RoPE : public Function {
public:

    std::shared_ptr<Variable> forward(const std::shared_ptr<Variable>& x) {
        throw std::invalid_argument("RoPE::forward requires cosine and sine variables");
    }

    std::shared_ptr<Variable> forward(const std::shared_ptr<Variable>& x,
        const std::shared_ptr<Variable>& C,
        const std::shared_ptr<Variable>& S) {

        const Tensor& x_tensor = x->data; // Input Tensor
        const Tensor& C_tensor = C->data; // Cos Component
        const Tensor& S_tensor = S->data; // Sin Component 

        // 1. d_k와 d_half 계산 (x의 마지막 축 차원)
        // NOTE: TensorData::shape()가 std::vector<size_t>를 반환하고
        // x.ndim()이 텐서의 랭크를 반환한다고 가정
        int d_k = static_cast<int>(x_tensor.shape()[x_tensor.ndim() - 1]);
        int d_half = d_k / 2;
        
        // 2. 텐서 분할 (x_A와 x_B)
        // NOTE: nb::split이 마지막 축을 기준으로 분할한다고 가정합니다.
        Tensor x_A = nb::split(x_tensor, 0, d_half); // 앞쪽 절반 (x_0)
        Tensor x_B = nb::split(x_tensor, 1, d_half); // 뒤쪽 절반 (x_1)

        // 3. 회전 행렬 적용 (term1, term2 계산)
        // [x_0 * C - x_1 * S]
        Tensor term1 = (x_A * C_tensor) - (x_B * S_tensor); 

        // [x_1 * C + x_0 * S]
        Tensor term2 = (x_B * C_tensor) + (x_A * S_tensor); 
        
        // 4. 결과 텐서 결합 (Concatenation)
        std::vector<Tensor> input;
        input.push_back(term1);
        input.push_back(term2);
        Tensor y = nb::concat(input, x_tensor.ndim() - 1); // 

        return Variable::create(y);
    }

    std::vector<Tensor> forward(const std::vector<Tensor>& xs) override {
        auto out = forward(
            Variable::create(xs[0]),
            Variable::create(xs[1]),
            Variable::create(xs[2]));
        return { out->data };
    }
};

// Function Wrapper
inline std::shared_ptr<Variable> rope(const std::shared_ptr<Variable>& x,
    const std::shared_ptr<Variable>& C,
    const std::shared_ptr<Variable>& S) {
    auto f = std::make_shared<RoPE>();
    return f->forward(x, C, S);
}

} // namespace bs
