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
// 2️⃣ 인코딩 (문자열 → 토큰 ID)
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
