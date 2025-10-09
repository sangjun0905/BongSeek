#pragma once

#include "Core.hpp"
#include "Softmax.hpp"
#include "Linear.hpp"
#include <cmath>      

namespace bs {

class SelfAttentionLayer : public Module {
private:
    // Q, K, V, O Projection Linear Layers (Module 상속)
    std::shared_ptr<Linear> Q_proj, K_proj, V_proj; 
    std::shared_ptr<Linear> O_proj; 

    double dk_root_; // 스케일링 상수 (1 / sqrt(d_k))
    int key_dim_;

public:
    // key_dim은 d_k(헤드 차원)이며, input_dim은 임베딩 차원입니다.
    SelfAttentionLayer(int input_dim, int key_dim)
        : key_dim_(key_dim), dk_root_(std::sqrt(static_cast<double>(key_dim)))
    {
        // 1. Linear Projection 초기화: input_dim -> key_dim
        Q_proj = std::make_shared<Linear>(input_dim, key_dim);
        K_proj = std::make_shared<Linear>(input_dim, key_dim);
        V_proj = std::make_shared<Linear>(input_dim, key_dim);
        
        // 2. Output Projection 초기화: key_dim -> input_dim
        O_proj = std::make_shared<Linear>(key_dim, input_dim); 

        // 3. Module들을 SelfAttentionLayer의 하위 Module로 등록하여 관리를 용이하게 합니다.
        add_module("Q_proj", Q_proj);
        add_module("K_proj", K_proj);
        add_module("V_proj", V_proj);
        add_module("O_proj", O_proj);
        
        // NOTE: 실제 추론 시에는 여기에 가중치를 로드하는 로직이 추가됩니다.
    }

    // Module::forward 인터페이스 구현: 입력 x를 Q, K, V 모두에 사용합니다 (Self-Attention).
    std::shared_ptr<Variable> forward(const std::shared_ptr<Variable>& x) override {
        // x의 shape은 (Batch, Seq, Input_Dim)을 가정합니다.
        
        // 1. Linear Projection (Q, K, V 변환)
        auto Q = (*Q_proj)(x); // Q: (B, S, Dk)
        auto K = (*K_proj)(x); // K: (B, S, Dk)
        auto V = (*V_proj)(x); // V: (B, S, Dk)

        // 2. Score 계산: Q @ K^T
        // K 전치: (B, S, Dk) -> (B, Dk, S) [마지막 두 축 전치]
        auto K_T = Variable::create(K->data.transpose()); 
        
        // scores = Q @ K^T: (B, S, Dk) @ (B, Dk, S) = (B, S, S)
        auto scores = matmul(Q, K_T); 

        // 3. 스케일링: scores / sqrt(d_k)
        // 스케일링 상수 (dk_root_의 역수)
        double scale_factor = 1.0 / dk_root_;
        
        // Mul 연산자 오버로딩을 사용하여 스케일링 (스칼라와 Variable 곱셈)
        // NOTE: core.hpp에 Variable과 스칼라의 연산이 구현되어 있다고 가정
        auto scaled_scores = mul(scores, Variable::create(nb::array({(TensorValueType)scale_factor})));


        // 4. Softmax 적용: Attention Weights
        // Softmax는 스코어의 마지막 축(Seq 축, axis=2)에 적용하여 합이 1이 되도록 합니다.
        auto attention_weights = softmax(scaled_scores, 2); 

        // 5. Context Vector 계산: Attention Weights @ V
        // context_vector = Attn_Weights @ V : (B, S, S) @ (B, S, Dk) = (B, S, Dk)
        auto context_vector = matmul(attention_weights, V);

        // 6. Output Projection (O)
        auto output = (*O_proj)(context_vector); // Output: (B, S, Input_Dim)

        return output;
    }
};

} // namespace bs
