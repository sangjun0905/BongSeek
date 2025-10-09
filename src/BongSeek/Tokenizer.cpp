#include "BongSeek/Tokenizer.hpp"

#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

bool Tokenizer::load(const std::string& path) {
    fs::path model(path);
    if (!model.empty() && fs::exists(model)) {
        model_path_ = fs::weakly_canonical(model).string();
        std::cout << "[Tokenizer] Loaded model from " << model_path_ << "\n";
        return true;
    }

    std::cerr << "[Tokenizer] Model not found: " << path << "\n";
    model_path_.clear();
    return false;
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    std::vector<int> tokens;
    tokens.reserve(text.size());
    for (unsigned char ch : text) {
        tokens.push_back(static_cast<int>(ch));
    }
    return tokens;
}

std::string Tokenizer::decode(const std::vector<int>& tokens) const {
    std::string text;
    text.reserve(tokens.size());
    for (int token : tokens) {
        if (token < 0 || token > 255) {
            text.push_back('?');
        } else {
            text.push_back(static_cast<char>(token));
        }
    }
    return text;
}
