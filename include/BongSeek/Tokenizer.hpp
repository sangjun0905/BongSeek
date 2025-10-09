#pragma once

#include <string>
#include <vector>

/**
 * Minimal tokenizer stub that performs byte-level encoding/decoding.
 *
 * The real project is expected to integrate with SentencePiece or another
 * subword tokenizer.  This lightweight placeholder keeps the runtime
 * utilities operational without external dependencies.
 */
class Tokenizer {
public:
    Tokenizer() = default;

    bool load(const std::string& path);
    std::vector<int> encode(const std::string& text) const;
    std::string decode(const std::vector<int>& tokens) const;

private:
    std::string model_path_;
};

