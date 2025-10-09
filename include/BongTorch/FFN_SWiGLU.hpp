#pragma once
#include "Core.hpp"
#include "Linear.hpp"
#include "SiLU.hpp"
#include "Module.hpp"

namespace bs {

class FFN_SWiGLU : public Module {
private:
    std::shared_ptr<Linear> gate_linear; 
    std::shared_ptr<Linear> value_linear; 
    std::shared_ptr<Linear> down_linear; 

public:
    FFN_SWiGLU() {};

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
        
        // 3. 원소별 곱셈 (SWiGLU 핵심)
        // Variable의 * 연산자 오버로딩 (Mul Function) 사용
        auto hidden_state = mul(activated_gate, value_output); 
        
        auto output = (*down_linear)(hidden_state);

        return output;
    }

    void loadWeights(std::istream& file, const MetadataMap& metadata){
        MetadataMap gate_linear_meta;
        MetadataMap value_linear_meta;
        MetadataMap down_linear_meta;


        for(auto& [key, value] : metadata) {
            if(key.compare(0, 12, "gate_linear.") == 0) {
                gate_linear_meta[key.substr(12)] = value; // "gate_linear." 제외
            } 
            else if (key.compare(0, 13, "value_linear.") == 0) {
                value_linear_meta[key.substr(13)] = value; // "value_linear." 제외
            } 
            else if (key.compare(0, 12, "down_linear.") == 0) {
                down_linear_meta[key.substr(12)] = value; // "down_linear." 제외
            }
        }

        gate_linear->loadWeights(file, gate_linear_meta);
        value_linear->loadWeights(file, value_linear_meta);
        down_linear->loadWeights(file, down_linear_meta);
    }
};

} // namespace bs
