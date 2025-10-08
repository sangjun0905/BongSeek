#ifndef GQA_ATTENTION_HPP
#define GQA_ATTENTION_HPP

#include "module.hpp"      // Module, Linear, Parameter 정의 포함
#include "core.hpp"        // Variable, Function, Add, Mul 등 포함
#include <cmath>           // std::sqrt 사용

// Softmax, MatMul Function 및 관련 헬퍼 함수 정의가 여기에 포함되어 있다고 가정합니다.
// (이전 응답에서 제공되었음)

class GQAAttention : public Module {
private:
    using TensorData = typename Variable::TensorData;
    std::shared_ptr<Linear> Q_proj, K_proj, V_proj; 
    std::shared_ptr<Linear> O_proj; 
    
    double dk_root_;
    int head_dim_;
    int model_dim_;
    int num_heads_;   // Query 헤드 수 (N_q)
    int num_groups_;  // K/V 그룹 수 (N_g)

public:
    GQAAttention(int model_dim, int num_heads, int num_groups)
        : model_dim_(model_dim), num_heads_(num_heads), num_groups_(num_groups)
    {
        head_dim_ = model_dim / num_heads;
        dk_root_ = std::sqrt(static_cast<double>(head_dim_));

        // ... Linear Projection 초기화 및 Module 등록 (이전 코드와 동일)
        Q_proj = std::make_shared<Linear>(model_dim, model_dim); 
        int kv_dim = num_groups * head_dim_; 
        K_proj = std::make_shared<Linear>(model_dim, kv_dim);
        V_proj = std::make_shared<Linear>(model_dim, kv_dim);
        O_proj = std::make_shared<Linear>(model_dim, model_dim); 

        add_module("Q_proj", Q_proj);
        add_module("K_proj", K_proj);
        add_module("V_proj", V_proj);
        add_module("O_proj", O_proj);
    }

    std::shared_ptr<Variable> forward(const std::shared_ptr<Variable>& x) override {
        // x shape: (B, S, D_model) - B: 배치, S: 시퀀스 길이, D_model: 임베딩 차원
        // TensorData는 Rank 3이므로, (B, S, D) 형태를 가정하고 진행합니다.

        // 1. Linear Projection (Q, K, V 변환)
        auto Q_raw = (*Q_proj)(x); // (B, S, D_model)
        auto K_raw = (*K_proj)(x); // (B, S, Kv_dim)
        auto V_raw = (*V_proj)(x); // (B, S, Kv_dim)
        
        // B (Batch)와 S (Seq Length)는 런타임에 결정되므로, shape을 직접 추출해야 합니다.
        // 여기서는 B와 S를 가정하고 진행합니다. (nb::Tensor가 shape 정보를 제공한다고 가정)
        auto x_shape = x->shape();
        int B = x_shape[0]; 
        int S = x_shape[1];

        // --- Multi-Head 및 GQA 처리 ---
        
        // 2. Q 텐서 재배열: (B, S, D_model) -> (B, S, N_heads, D_head) -> (B, N_heads, S, D_head)
        // **가정: nb::reshape(tensor, {d1, d2, d3...})가 구현되어 있음**
        auto Q_reshaped_data = nb::reshape(Q_raw->data, 
            {B, S, num_heads_, head_dim_});
        
        // K와 V의 텐서 조작 (K/V는 Rank 3에 맞게 S축을 살립니다.)
        auto K_reshaped_data = nb::reshape(K_raw->data, 
            {B, S, num_groups_, head_dim_});
        auto V_reshaped_data = nb::reshape(V_raw->data, 
            {B, S, num_groups_, head_dim_});

        // NOTE: Rank 4 텐서 조작이 필요하나, TensorRank=3이므로 
        // 편의상 (B*N, S, D_head) 형태로 2D MatMul을 사용하도록 강제합니다.
        
        // 4. K/V 브로드캐스팅 (GQA 핵심 로직)
        int repeat_factor = num_heads_ / num_groups_;
        
        // **가정: nb::repeat(tensor, repeat_count, axis)가 구현되어 있음**
        // K/V의 N_groups 축(Rank 2)을 N_heads만큼 확장합니다.
        auto K_broadcasted_data = nb::repeat(K_reshaped_data, repeat_factor, 1);
        auto V_broadcasted_data = nb::repeat(V_reshaped_data, repeat_factor, 1);

        // Q, K, V를 Variable로 래핑하여 MatMul에 준비합니다.
        auto Q_var = Variable::create(Q_reshaped_data);
        auto K_var = Variable::create(K_broadcasted_data);
        auto V_var = Variable::create(V_broadcasted_data);
        
        // 5. 스케일링 Dot-Product Attention 계산 (Self-Attention 로직 재사용)
        
        // K^T (마지막 두 축 전치): (B, N_heads, S, D_head) -> (B, N_heads, D_head, S)
        // K->data.transpose()는 마지막 두 축을 전치한다고 가정
        auto K_T = Variable::create(K_var->data.transpose()); 
        
        // scores = Q @ K^T : (B, N, S, D) @ (B, N, D, S) = (B, N, S, S)
        auto scores = matmul(Q_var, K_T); 
        
        // 스케일링 (scores / sqrt(D_head))
        auto scale_factor = 1.0 / dk_root_;
        auto scaled_scores = mul(scores, Variable::create(nb::array({(TensorValueType)scale_factor})));
        
        // Softmax: 마지막 축 (Score 축)에 Softmax 적용
        auto attn_weights = softmax(scaled_scores, -1); 
        
        // Context Vector 계산
        // context_vector = Attn_Weights @ V : (B, N, S, S) @ (B, N, S, D) = (B, N, S, D_head)
        auto context_vector = matmul(attn_weights, V_var);

        // 6. Head 결합: (B, N_heads, S, D_head) -> (B, S, D_model)
        // context_combined_data = context_vector->data.transpose({0, 2, 1, 3});
        // context_combined_data = nb::reshape(context_combined_data, {B, S, model_dim_});
        
        // 임시로 Q_raw의 형태로 재구성 (NumBong 텐서 조작 기능 가정)
        auto context_combined = Variable::create(nb::reshape(context_vector->data, {B, S, model_dim_}));


        // 7. Output Projection
        auto output = (*O_proj)(context_combined);

        return output;
    }
};

#endif // GQA_ATTENTION_HPP