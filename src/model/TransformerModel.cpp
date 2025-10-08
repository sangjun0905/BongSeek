#include "TransformerModel.hpp"
#include "LiquidBlock.hpp"
#include <iostream>
#include <iostream>
#include <numeric>

void TransformerModel::init(const ModelConfig& cfg, WeightLoader& loader) {
    config = cfg;
    blocks.clear();

    std::cout << "[TransformerModel] LFM2 구조 생성 중..." << std::endl;

    // Embedding
    embedding = loader.get("model.embed_tokens.weight");
    std::cout << "Embedding loaded (" << config.vocab_size << "×" << config.hidden_size << ")\n";

    // Blocks
    for (int i = 0; i < config.num_hidden_layers; ++i) {
        LiquidBlock block;
        block.init(i, loader);
        blocks.push_back(std::move(block));
    }

    // LM Head
    lm_head = loader.get("lm_head.weight");

    std::cout << "[TransformerModel] 초기화 완료 (" 
              << config.num_hidden_layers << " layers)\n";
}

#include "TransformerModel.hpp"
#include "LiquidBlock.hpp"
#include <iostream>
#include <iostream>
#include <numeric>

void TransformerModel::init(const ModelConfig& cfg, WeightLoader& loader) {
    config = cfg;
    blocks.clear();

    std::cout << "[TransformerModel] LFM2 구조 생성 중..." << std::endl;

    // Embedding
    embedding = loader.get("model.embed_tokens.weight");
    std::cout << "Embedding loaded (" << config.vocab_size << "×" << config.hidden_size << ")\n";

    // Blocks
    for (int i = 0; i < config.num_hidden_layers; ++i) {
        LiquidBlock block;
        block.init(i, loader);
        blocks.push_back(std::move(block));
    }

    // LM Head
    lm_head = loader.get("lm_head.weight");

    std::cout << "[TransformerModel] 초기화 완료 (" 
              << config.num_hidden_layers << " layers)\n";
}

