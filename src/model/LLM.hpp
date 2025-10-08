#pragma once
#include <string>
#include "ModelConfig.hpp"
#include "Tokenizer.hpp"
#include "WeightLoader.hpp"
#include "TransformerModel.hpp"

class LLM {
private:
    ModelConfig config;
    Tokenizer tokenizer;
    TransformerModel model;
    WeightLoader loader;

public:
    bool load(const std::string& model_name);
    std::string generate(const std::string& prompt, int max_tokens);
};
