#pragma once

#include "Core.hpp"

namespace bs {

class RMSNormFunction : public Function { 
private:
    const nb::BFloat16 epsilon_{nb::BFloat16(1e-5f)};

public:
    // Forward: y = x * rsqrt(mean(x^2) + eps) * gamma
    // xs[0] = x (입력 텐서), xs[1] = gamma (가중치 텐서)
    std::vector<Tensor> forward(const std::vector<Tensor>& xs) override {
        // nb::Array 대신 TensorData (Core.hpp의 using에 따름)로 타입을 통일합니다.
        const Tensor& x = xs[0]; 
        const Tensor& gamma = xs[1]; 

        // 1. x^2
        Tensor x1 = x ^ 2.0; 
        
        // 2. mean(x^2)        
        Tensor mean_x1 = nb::mean(x1); 
        
        // 3. rsqrt(mean(x^2) + epsilon)
        Tensor rrms = nb::rsqrt(mean_x1 + epsilon_); 
        
        // 4. y = x * rrms * gamma
        Tensor y = x * rrms * gamma; 

        return { y };
    }
};

// Function Wrapper (bs::rms_norm)
inline std::shared_ptr<Variable> rms_norm(const std::shared_ptr<Variable>& x, const std::shared_ptr<Variable>& g) {
    auto f = std::make_shared<RMSNormFunction>();
    return (*f)({x, g});
}

} // namespace bs
