// FFN_SWiGLU.hpp
#pragma once
#include "Module.hpp"
#include "Linear.hpp"
#include "SiLU.hpp" // silu Function 사용

namespace bs {

class FFN_SWiGLU : public Module {
private:
    // W_1 변환을 담당 (업스케일링)
    std::shared_ptr<Linear> gate_linear; 
    // W_2 변환을 담당 (업스케일링)
    std::shared_ptr<Linear> value_linear; 
    // W_3 변환을 담당 (다운스케일링)
    std::shared_ptr<Linear> down_linear; 

    string name;

public:
    FFN_SWiGLU(const string& prefix, int embed_dim, int hidden_dim) {
        // Linear 초기화 및 Module 등록 (편향 사용 여부는 설계에 따라 다름)
        name = prefix;

        gate_linear  = std::make_shared<Linear>(embed_dim, hidden_dim, false);
        value_linear = std::make_shared<Linear>(embed_dim, hidden_dim, false);
        down_linear  = std::make_shared<Linear>(hidden_dim, embed_dim, false);
        
        add_module("gate_linear", gate_linear);
        add_module("value_linear", value_linear);
        add_module("down_linear", down_linear);
    }

    std::shared_ptr<Variable> forward(const std::shared_ptr<Variable>& x) override {
        // 1. Gate 경로 (SiLU 적용 경로)
        auto gate_output = (*gate_linear)(x); // x * W_1
        auto activated_gate = silu(gate_output); // SiLU(x * W_1)
        
        // 2. Value 경로
        auto value_output = (*value_linear)(x); // x * W_2
        
        // 3. 원소별 곱셈 (SWiGLU 핵심)
        // Variable의 * 연산자 오버로딩 (Mul Function) 사용
        auto hidden_state = activated_gate * value_output; 
        
        // 4. Down Projection
        auto output = (*down_linear)(hidden_state); // (SiLU(...) * Value) * W_3

        return output;
    }
};

} // namespace bs