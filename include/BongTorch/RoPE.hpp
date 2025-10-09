#pragma once 

#include "Core.hpp"
#include <memory>

namespace bs {

class RoPE : public Function {
public:

    std::vector<TensorData> forward(const std::vector<TensorData>& xs) override {
        const TensorData& x = xs[0]; // Input Tensor
        const TensorData& C = xs[1]; // Cos Component
        const TensorData& S = xs[2]; // Sin Component 

        // 1. d_k와 d_half 계산 (x의 마지막 축 차원)
        // NOTE: TensorData::shape()가 std::vector<size_t>를 반환하고
        // x.ndim()이 텐서의 랭크를 반환한다고 가정
        int d_k = static_cast<int>(x.shape()[x.ndim() - 1]);
        int d_half = d_k / 2;
        
        // 2. 텐서 분할 (x_A와 x_B)
        // NOTE: nb::split이 마지막 축을 기준으로 분할한다고 가정합니다.
        TensorData x_A = nb::split(x, 0, d_half); // 앞쪽 절반 (x_0)
        TensorData x_B = nb::split(x, 1, d_half); // 뒤쪽 절반 (x_1)

        // 3. 회전 행렬 적용 (term1, term2 계산)
        // [x_0 * C - x_1 * S]
        TensorData term1 = (x_A * C) - (x_B * S); 

        // [x_1 * C + x_0 * S]
        TensorData term2 = (x_B * C) + (x_A * S); 
        
        // 4. 결과 텐서 결합 (Concatenation)
        TensorData y = nb::concat({ term1, term2 }, x.ndim() - 1); // 

        return { y };
    }
};

// Function Wrapper
inline std::shared_ptr<Variable> rope(const std::shared_ptr<Variable>& x,
    const std::shared_ptr<Variable>& C,
    const std::shared_ptr<Variable>& S) {
    auto f = std::make_shared<RoPE>();
    // (*f) 오버로딩을 통해 Function 호출
    auto outs = (*f)(std::vector<std::shared_ptr<Variable>>{x, C, S});
    return outs[0];
}

} // namespace bs