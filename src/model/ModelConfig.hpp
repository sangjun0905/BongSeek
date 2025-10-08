#pragma once
#include <string>

class ModelConfig {
public:
    int vocab_size;
    int hidden_size;
    int num_hidden_layers;
    int num_attention_heads;
    int bos_token_id;
    int eos_token_id;

    bool load(const std::string& path);
};
