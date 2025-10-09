#pragma once
#include "Core.hpp"
#include "Linear.hpp"
#include "SiLU.hpp"

namespace bs {

class FFN_SWiGLU : public Module {
private:
    std::shared_ptr<Linear> gate_linear; 
    std::shared_ptr<Linear> value_linear; 
    std::shared_ptr<Linear> down_linear; 

public:
    FFN_SWiGLU(int embed_dim, int hidden_dim) {
        gate_linear  = std::make_shared<Linear>(embed_dim, hidden_dim, false);
        value_linear = std::make_shared<Linear>(embed_dim, hidden_dim, false);
        down_linear  = std::make_shared<Linear>(hidden_dim, embed_dim, false);
        
        add_module("gate_linear", gate_linear);
        add_module("value_linear", value_linear);
        add_module("down_linear", down_linear);
    }

    std::shared_ptr<Variable> forward(const std::shared_ptr<Variable>& x) override {
        auto gate_output = (*gate_linear)(x);
        auto activated_gate = silu(gate_output);
        
        auto value_output = (*value_linear)(x);
        
        auto hidden_state = mul(activated_gate, value_output); 
        
        auto output = (*down_linear)(hidden_state);

        return output;
    }

    // Getter methods for testing
    std::shared_ptr<Linear> get_gate_linear() const { return gate_linear; }
    std::shared_ptr<Linear> get_value_linear() const { return value_linear; }
    std::shared_ptr<Linear> get_down_linear() const { return down_linear; }
};

} // namespace bs
