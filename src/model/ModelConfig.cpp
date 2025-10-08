#include "ModelConfig.hpp"
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

bool ModelConfig::load(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "[ModelConfig] 파일 열기 실패: " << path << std::endl;
        return false;
    }

    json j;
    file >> j;

    vocab_size = j.value("vocab_size", 0);
    hidden_size = j.value("hidden_size", 0);
    num_hidden_layers = j.value("num_hidden_layers", 0);
    num_attention_heads = j.value("num_attention_heads", 0);
    bos_token_id = j.value("bos_token_id", 0);
    eos_token_id = j.value("eos_token_id", 0);

    std::cout << "[ModelConfig] 로드 완료" << std::endl;
    return true;
}
