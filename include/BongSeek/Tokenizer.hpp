#pragma once

#include <filesystem>
#include <memory>
#include <regex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <nlohmann/json.hpp>
#include <sentencepiece_processor.h>

/**
 * Hybrid tokenizer wrapper. Prefers SentencePiece models (*.model / *.spm)
 * when available, and otherwise falls back to the HuggingFace GPT-style BPE
 * pipeline defined by tokenizer.json.
 */
class Tokenizer {
public:
    Tokenizer() = default;

    bool load(const std::string& path);
    std::vector<int> encode(const std::string& text) const;
    std::string decode(const std::vector<int>& tokens) const;

    int token_to_id(const std::string& token) const;
    std::string id_to_token(int id) const;

    int bos_token_id() const { return bos_id_; }
    int eos_token_id() const { return eos_id_; }
    int pad_token_id() const { return pad_id_; }
    bool is_special_id(int id) const;

private:
    enum class Backend {
        None,
        SentencePiece,
        GPTBPE,
    };

    // Common helpers
    static std::string extract_token_string(const nlohmann::json& value);
    int resolve_token_id(const nlohmann::json& value) const;

    // SentencePiece helpers
    bool load_sentencepiece_model(const std::filesystem::path& model_path);
    void infer_special_tokens(const std::filesystem::path& model_path);
    void apply_special_tokens_map(const std::filesystem::path& map_path);
    void apply_tokenizer_config(const std::filesystem::path& config_path);

    // GPT-BPE helpers
    bool load_bpe_from_json(const std::filesystem::path& json_path);
    void build_byte_maps();
    void load_special_tokens(const nlohmann::json& tokenizer_json);
    void load_additional_metadata(const std::filesystem::path& base_dir);
    void encode_segment_bpe(const std::string& text, std::vector<int>& output) const;
    void encode_token_bpe(const std::string& token, std::vector<int>& output) const;
    std::vector<std::string> apply_bpe(const std::string& token) const;
    static std::string make_pair_key(const std::string& first, const std::string& second);

    Backend backend_{Backend::None};
    bool loaded_{false};

    // SentencePiece backend
    std::shared_ptr<sentencepiece::SentencePieceProcessor> processor_;

    // GPT-BPE backend data
    std::unordered_map<std::string, int> vocab_;
    std::vector<std::string> id_to_token_;
    std::unordered_map<std::string, std::size_t> merge_ranks_;
    std::regex pattern_;
    std::unordered_map<uint32_t, std::string> byte_encoder_;
    std::unordered_map<std::string, uint32_t> byte_decoder_;
    std::unordered_map<std::string, int> special_tokens_;
    std::vector<std::pair<std::string, int>> special_tokens_sorted_;
    std::unordered_set<int> special_token_ids_;
    mutable std::unordered_map<std::string, std::vector<std::string>> bpe_cache_;

    int bos_id_{-1};
    int eos_id_{-1};
    int pad_id_{-1};
};
