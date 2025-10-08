#pragma once
#include <vector>
#include <string>
#include "ModelConfig.hpp"
#include "WeightLoader.hpp"
#include "LiquidBlock.hpp"

class TransformerModel {
private:
    ModelConfig config;

    // === 모델 파라미터 ===
    std::vector<float> embedding;        // 임베딩 가중치 (vocab_size × hidden_size)
    std::vector<float> lm_head;          // 출력 projection (hidden_size × vocab_size)
    std::vector<LiquidBlock> blocks;     // 30개 블록 저장

public:
    void init(const ModelConfig& cfg, WeightLoader& loader);
    std::vector<float> forward(const std::vector<int>& input_ids);
    int sample(const std::vector<float>& logits);
};
