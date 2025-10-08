#pragma once
#include <string>
#include <vector>
#include <sentencepiece_processor.h>

class Tokenizer {
private:
    sentencepiece::SentencePieceProcessor sp;  // SentencePiece 엔진

public:
    bool load(const std::string& model_path);  // tokenizer.model 로드
    std::vector<int> encode(const std::string& text);
    std::string decode(const std::vector<int>& ids);
};

















/*#pragma once
#include <string>
#include <unordered_map>
#include <vector>

class Tokenizer {
private:
    std::unordered_map<std::string, int> vocab;     // 단어 → ID
    std::unordered_map<int, std::string> inv_vocab; // ID → 단어
    std::vector<std::pair<std::string, std::string>> merges; // BPE 규칙

    std::unordered_map<std::string, int> added_specials; // <s>, </s> 등

public:
    bool load(const std::string& path);
    std::vector<int> encode(const std::string& text);
    std::string decode(const std::vector<int>& ids);
};
*/