#pragma once

#include "Core.hpp"
#include <memory>
#include <stdexcept>
#include <vector>

namespace bs {

class EmbeddingFunction : public Function {
// private 멤버에서 역전파용 flat_indices_ 제거
private:
    std::size_t batch_size_{0};
    std::size_t seq_len_{0};
    std::size_t embedding_dim_{0};
    std::size_t vocab_size_{0};
    
public:
    std::vector<TensorData> forward(const std::vector<TensorData>& xs) override {
        const auto& indices = xs[0];
        const auto& weight = xs[1];

        // 1. Shape 검증 및 정보 추출 (통일된 .shape() 사용)
        auto idx_shape = indices.shape();
        if (idx_shape.size() != 3 || idx_shape[2] != 1) { // ndim 검증 추가
            throw std::runtime_error("Indices tensor must have shape (B, S, 1).");
        }
        
        auto weight_shape = weight.shape();
        batch_size_ = idx_shape[0];
        seq_len_ = idx_shape[1];
        vocab_size_ = weight_shape[0];
        embedding_dim_ = weight_shape[1];

        TensorShape out_shape = {batch_size_, seq_len_, embedding_dim_};
        TensorData output(out_shape);

        // 2. 임베딩 룩업 (추론 로직)
        for (std::size_t b = 0; b < batch_size_; ++b) {
            for (std::size_t s = 0; s < seq_len_; ++s) {
                // 인덱스 추출 및 검증
                float raw = indices(b, s, 0); 
                if (raw < 0.0f) {
                    throw std::runtime_error("Embedding index must be non-negative.");
                }
                std::size_t idx = static_cast<std::size_t>(raw);
                if (idx >= vocab_size_) {
                    throw std::runtime_error("Embedding index out of range.");
                }

                // 3. 룩업 (데이터 복사)
                for (std::size_t d = 0; d < embedding_dim_; ++d) {
                    // weight(idx, d, 0)에서 값을 읽어와 output(b, s, d)에 씁니다.
                    output(b, s, d) = weight(idx, d, 0);
                }
            }
        }

        return {output};
    }
    
    // 추론 전용이므로 backward는 미완성 상태를 유지
    std::vector<std::shared_ptr<Variable>> backward(const std::vector<std::shared_ptr<Variable>>& gys) override {
        // dL/d(indices)와 dL/d(weight) 기울기 반환
        return { nullptr, nullptr }; 
    }
};

// Function Wrapper (외부 API)
inline std::shared_ptr<Variable> embedding_lookup(const std::shared_ptr<Variable>& indices,
                                                  const std::shared_ptr<Variable>& weight) {
    auto f = std::make_shared<EmbeddingFunction>();
    auto outs = (*f)(std::vector<std::shared_ptr<Variable>>{indices, weight});
    return outs[0];
}

}