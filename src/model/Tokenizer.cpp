#include "Tokenizer.hpp"
#include <iostream>

// -----------------------------------------------------
// 1️⃣ SentencePiece 모델 로드
// -----------------------------------------------------
bool Tokenizer::load(const std::string& model_path) {
    auto status = sp.Load(model_path);
    if (!status.ok()) {
        std::cerr << "[Tokenizer] SentencePiece 모델 로드 실패: " 
                  << status.ToString() << std::endl;
        return false;
    }

    std::cout << "[Tokenizer] SentencePiece 모델 로드 완료 (" 
              << model_path << ")\n";
    return true;
}

// -----------------------------------------------------
// 2️⃣ 인코딩 (문자열 → 토큰 ID 벡터)
// -----------------------------------------------------
std::vector<int> Tokenizer::encode(const std::string& text) {
    std::vector<int> ids;
    auto status = sp.Encode(text, &ids);
    if (!status.ok()) {
        std::cerr << "[Tokenizer] 인코딩 실패: " 
                  << status.ToString() << std::endl;
        return {};
    }
    return ids;
}

// -----------------------------------------------------
// 3️⃣ 디코딩 (토큰 ID → 문자열)
// -----------------------------------------------------
std::string Tokenizer::decode(const std::vector<int>& ids) {
    std::string text;
    auto status = sp.Decode(ids, &text);
    if (!status.ok()) {
        std::cerr << "[Tokenizer] 디코딩 실패: " 
                  << status.ToString() << std::endl;
        return "";
    }
    return text;
}



















/*#include "Tokenizer.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <nlohmann/json.hpp>
#include <algorithm>

using json = nlohmann::json;

// -----------------------------------------------------
// 1️⃣ tokenizer.json 로드
// -----------------------------------------------------
bool Tokenizer::load(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "[Tokenizer] 파일 열기 실패: " << path << std::endl;
        return false;
    }

    json j;
    file >> j;

    // Vocab 로드
    if (j.contains("model") && j["model"].contains("vocab")) {
        for (auto& [token, id] : j["model"]["vocab"].items()) {
            vocab[token] = id.get<int>();
            inv_vocab[id.get<int>()] = token;
        }
        std::cout << "[Tokenizer] Vocab 로드 완료 (" << vocab.size() << "개)\n";
    }

    // Merges 로드
    if (j["model"].contains("merges")) {
        for (auto& m : j["model"]["merges"]) {
            std::istringstream iss(m.get<std::string>());
            std::string a, b;
            iss >> a >> b;
            merges.push_back({a, b});
        }
        std::cout << "[Tokenizer] Merges 규칙 로드 완료 (" << merges.size() << "개)\n";
    }

    // Added tokens
    if (j.contains("added_tokens")) {
        for (auto& t : j["added_tokens"]) {
            if (t.contains("content") && t.contains("id"))
                added_specials[t["content"].get<std::string>()] = t["id"].get<int>();
        }
        std::cout << "[Tokenizer] 스페셜 토큰 로드 완료 (" << added_specials.size() << "개)\n";
    }

    return true;
}

// -----------------------------------------------------
// 2️⃣ BPE 인코딩 (간이 버전)
// -----------------------------------------------------
std::vector<int> Tokenizer::encode(const std::string& text) {
    std::vector<int> ids;

    // 간단한 단어 단위 분리 (실제는 SentencePiece처럼 ▁ 기반)
    std::istringstream iss(text);
    std::string word;

    while (iss >> word) {
        std::string token = "▁" + word; // SentencePiece prefix
        if (vocab.count(token)) {
            ids.push_back(vocab[token]);
        } else {
            // BPE 병합 적용 (간단히 시연용)
            std::string current = token;
            for (auto& merge : merges) {
                std::string pattern = merge.first + merge.second;
                size_t pos = current.find(pattern);
                if (pos != std::string::npos) {
                    current.replace(pos, pattern.size(), pattern);
                }
            }
            if (vocab.count(current))
                ids.push_back(vocab[current]);
            else
                ids.push_back(vocab.size()); // unknown token
        }
    }

    return ids;
}

// -----------------------------------------------------
// 3️⃣ 디코딩
// -----------------------------------------------------
std::string Tokenizer::decode(const std::vector<int>& ids) {
    std::string text;
    for (int id : ids) {
        if (inv_vocab.count(id)) {
            std::string token = inv_vocab[id];
            if (token.rfind("▁", 0) == 0) // SentencePiece-style 공백
                text += " " + token.substr(3);
            else
                text += token;
        }
    }
    return text;
}
