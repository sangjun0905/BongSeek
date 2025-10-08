#pragma once

#include "Core.hpp"

namespace bs {

class RMSNormFunction : public Function { 
private:
    const double epsilon_ = 1e-5; 

public:
    // Forward: y = x * rsqrt(mean(x^2) + eps) * gamma
    // xs[0] = x (입력 텐서), xs[1] = gamma (가중치 텐서)
    std::vector<TensorData> forward(const std::vector<TensorData>& xs) override {
        // nb::Array 대신 TensorData (Core.hpp의 using에 따름)로 타입을 통일합니다.
        const TensorData& x = xs[0]; 
        const TensorData& gamma = xs[1]; 

        // 1. x^2
        TensorData x1 = x ^ 2.0; 
        
        // 2. mean(x^2)        
        TensorData mean_x1 = nb::mean(x1); 
        
        // 3. rsqrt(mean(x^2) + epsilon)
        TensorData rrms = nb::rsqrt(mean_x1 + epsilon_); 
        
        // 4. y = x * rrms * gamma
        TensorData y = x * rrms * gamma; 

        return { y };
    }
};

// Function Wrapper (bs::rms_norm)
inline std::shared_ptr<Variable> rms_norm(const std::shared_ptr<Variable>& x, const std::shared_ptr<Variable>& g) {
    auto f = std::make_shared<RMSNormFunction>();
    auto outs = (*f)(std::vector<std::shared_ptr<Variable>>{x, g}); 
    return outs[0];
}

} // namespace bs