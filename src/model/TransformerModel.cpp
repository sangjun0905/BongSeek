#include "TransformerModel.hpp"
#include "LiquidBlock.hpp"
#include <algorithm>
#include <iostream>

void TransformerModel::init(const ModelConfig& cfg, WeightLoader& loader) {
    config = cfg;
    blocks.clear();

    std::cout << "[TransformerModel] LFM2 구조 구성 중..." << std::endl;

    embedding = loader.get("model.embed_tokens.weight");
    std::cout << "Embedding loaded (" << config.vocab_size
              << "×" << config.hidden_size << ")\n";

    for (int i = 0; i < config.num_hidden_layers; ++i) {
        LiquidBlock block;
        block.init(i, loader);
        blocks.push_back(std::move(block));
    }

    lm_head = loader.get("lm_head.weight");

    std::cout << "[TransformerModel] 초기화 완료 ("
              << config.num_hidden_layers << " layers)\n";
}

std::vector<float> TransformerModel::forward(const std::vector<int>& input_ids) {
    if (config.vocab_size <= 0) return {};

    std::vector<float> logits(static_cast<size_t>(config.vocab_size), 0.0f);
    if (!lm_head.empty()) {
        size_t copy_count = std::min(logits.size(), lm_head.size());
        std::copy_n(lm_head.begin(), copy_count, logits.begin());
    }

    if (!input_ids.empty()) {
        size_t idx = static_cast<size_t>(input_ids.back()) % logits.size();
        logits[idx] += 1.0f; // 간단한 자극값
    }

    return logits;
}

int TransformerModel::sample(const std::vector<float>& logits) {
    if (logits.empty()) return 0;
    auto max_it = std::max_element(logits.begin(), logits.end());
    return static_cast<int>(std::distance(logits.begin(), max_it));
}
