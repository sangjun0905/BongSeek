#include "BongSeek/Tokenizer.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <optional>
#include <cctype>

using json = nlohmann::json;
namespace fs = std::filesystem;

namespace {

constexpr char kPairSeparator = '\x1F';

std::vector<std::string> split_utf8(const std::string& text) {
    std::vector<std::string> chars;
    for (std::size_t i = 0; i < text.size();) {
        const unsigned char byte = static_cast<unsigned char>(text[i]);
        std::size_t length = 1;
        if ((byte & 0x80) == 0x00) {
            length = 1;
        } else if ((byte & 0xE0) == 0xC0) {
            length = 2;
        } else if ((byte & 0xF0) == 0xE0) {
            length = 3;
        } else if ((byte & 0xF8) == 0xF0) {
            length = 4;
        } else {
            length = 1;
        }
        if (i + length > text.size()) {
            length = text.size() - i;
        }
        chars.emplace_back(text.substr(i, length));
        i += length;
    }
    return chars;
}

std::string utf8_encode(char32_t codepoint) {
    std::string out;
    if (codepoint <= 0x7F) {
        out.push_back(static_cast<char>(codepoint));
    } else if (codepoint <= 0x7FF) {
        out.push_back(static_cast<char>(0xC0 | ((codepoint >> 6) & 0x1F)));
        out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    } else if (codepoint <= 0xFFFF) {
        out.push_back(static_cast<char>(0xE0 | ((codepoint >> 12) & 0x0F)));
        out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | ((codepoint >> 18) & 0x07)));
        out.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    }
    return out;
}

int piece_to_id_guarded(const sentencepiece::SentencePieceProcessor& proc,
                        const std::string& token) {
    const int unk_id = proc.unk_id();
    const int id = proc.PieceToId(token);
    if (id == unk_id && token != proc.IdToPiece(unk_id)) {
        return -1;
    }
    return id;
}

std::optional<fs::path> locate_within_directory(const fs::path& directory,
                                                const std::vector<std::string>& names) {
    for (const auto& name : names) {
        const fs::path candidate = directory / name;
        if (fs::exists(candidate)) {
            return candidate;
        }
    }
    return std::nullopt;
}

bool is_json_file(const fs::path& path) {
    if (!fs::exists(path) || !fs::is_regular_file(path)) {
        return false;
    }
    const auto ext = path.extension().string();
    if (!ext.empty()) {
        return ext == ".json";
    }
    std::ifstream in(path);
    if (!in) {
        return false;
    }
    char c = '\0';
    while (in.get(c)) {
        if (!std::isspace(static_cast<unsigned char>(c))) {
            return c == '{';
        }
    }
    return false;
}

bool is_sentencepiece_model(const fs::path& path) {
    const auto ext = path.extension().string();
    return ext == ".model" || ext == ".spm" || ext == ".proto";
}

} // namespace

bool Tokenizer::load(const std::string& path) {
    backend_ = Backend::None;
    loaded_ = false;
    processor_.reset();
    vocab_.clear();
    id_to_token_.clear();
    merge_ranks_.clear();
    special_tokens_.clear();
    special_tokens_sorted_.clear();
    special_token_ids_.clear();
    bpe_cache_.clear();
    bos_id_ = eos_id_ = pad_id_ = -1;

    fs::path model_path(path);
    if (model_path.empty()) {
        std::cerr << "[Tokenizer] Empty tokenizer path provided.\n";
        return false;
    }

    if (fs::is_directory(model_path)) {
        static const std::vector<std::string> model_candidates = {
            "tokenizer.model",
            "spiece.model",
            "tokenizer.spm",
            "tokenizer.proto"
        };
        static const std::vector<std::string> json_candidates = {
            "tokenizer.json"
        };
        if (auto found = locate_within_directory(model_path, model_candidates)) {
            model_path = *found;
        } else if (auto json_found = locate_within_directory(model_path, json_candidates)) {
            model_path = *json_found;
        } else {
            std::cerr << "[Tokenizer] Directory " << path
                      << " does not contain a supported tokenizer file.\n";
            return false;
        }
    }

    if (!fs::exists(model_path)) {
        std::cerr << "[Tokenizer] Tokenizer file does not exist: " << model_path << '\n';
        return false;
    }

    bool success = false;
    if (is_sentencepiece_model(model_path)) {
        success = load_sentencepiece_model(model_path);
    } else if (is_json_file(model_path)) {
        success = load_bpe_from_json(model_path);
    } else {
        // Attempt to detect JSON by content if no known extension.
        if (is_json_file(model_path)) {
            success = load_bpe_from_json(model_path);
        } else {
            success = load_sentencepiece_model(model_path);
        }
    }

    if (!success) {
        std::cerr << "[Tokenizer] Failed to initialise tokenizer from " << model_path << '\n';
        backend_ = Backend::None;
        loaded_ = false;
    }

    return success;
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    if (!loaded_) {
        return {};
    }

    switch (backend_) {
        case Backend::SentencePiece: {
            std::vector<int> ids;
            const auto status = processor_->Encode(text, &ids);
            if (!status.ok()) {
                std::cerr << "[Tokenizer] Encode failed: " << status.ToString() << '\n';
                return {};
            }
            return ids;
        }
        case Backend::GPTBPE: {
            if (text.empty()) {
                return {};
            }
            std::vector<int> tokens;

            std::size_t pos = 0;
            while (pos < text.size()) {
                bool matched_special = false;
                for (const auto& [token, id] : special_tokens_sorted_) {
                    if (token.empty()) {
                        continue;
                    }
                    if (pos + token.size() <= text.size() &&
                        text.compare(pos, token.size(), token) == 0) {
                        tokens.push_back(id);
                        pos += token.size();
                        matched_special = true;
                        break;
                    }
                }
                if (matched_special) {
                    continue;
                }

                std::size_t next_special = text.size();
                for (const auto& [token, _] : special_tokens_sorted_) {
                    if (token.empty()) {
                        continue;
                    }
                    const auto found = text.find(token, pos);
                    if (found != std::string::npos) {
                        next_special = std::min(next_special, found);
                    }
                }

                const std::string segment = text.substr(pos, next_special - pos);
                encode_segment_bpe(segment, tokens);
                pos = next_special;
            }

            return tokens;
        }
        case Backend::None:
        default:
            return {};
    }
}

std::string Tokenizer::decode(const std::vector<int>& tokens) const {
    if (!loaded_ || tokens.empty()) {
        return {};
    }

    switch (backend_) {
        case Backend::SentencePiece: {
            std::string text;
            const auto status = processor_->Decode(tokens, &text);
            if (!status.ok()) {
                std::cerr << "[Tokenizer] Decode failed: " << status.ToString() << '\n';
                return {};
            }
            return text;
        }
        case Backend::GPTBPE: {
            std::string result;
            for (int id : tokens) {
                if (id < 0 || static_cast<std::size_t>(id) >= id_to_token_.size()) {
                    continue;
                }
                const std::string& piece = id_to_token_[static_cast<std::size_t>(id)];
                if (piece.empty()) {
                    continue;
                }
                if (special_token_ids_.count(id) > 0) {
                    result.append(piece);
                    continue;
                }
                const auto chars = split_utf8(piece);
                for (const auto& ch : chars) {
                    const auto it = byte_decoder_.find(ch);
                    if (it == byte_decoder_.end()) {
                        result.append(ch);
                    } else {
                        result.push_back(static_cast<char>(it->second));
                    }
                }
            }
            return result;
        }
        case Backend::None:
        default:
            return {};
    }
}

int Tokenizer::token_to_id(const std::string& token) const {
    if (!loaded_) {
        return -1;
    }

    switch (backend_) {
        case Backend::SentencePiece:
            return processor_ ? piece_to_id_guarded(*processor_, token) : -1;
        case Backend::GPTBPE: {
            auto it = vocab_.find(token);
            return it == vocab_.end() ? -1 : it->second;
        }
        case Backend::None:
        default:
            return -1;
    }
}

std::string Tokenizer::id_to_token(int id) const {
    if (!loaded_ || id < 0) {
        return {};
    }

    switch (backend_) {
        case Backend::SentencePiece:
            return (processor_ && id < processor_->GetPieceSize())
                ? processor_->IdToPiece(id)
                : std::string{};
        case Backend::GPTBPE:
            return static_cast<std::size_t>(id) < id_to_token_.size()
                ? id_to_token_[static_cast<std::size_t>(id)]
                : std::string{};
        case Backend::None:
        default:
            return {};
    }
}

bool Tokenizer::is_special_id(int id) const {
    if (id < 0) {
        return true;
    }
    if (id == bos_id_ || id == eos_id_ || id == pad_id_) {
        return true;
    }
    if (special_token_ids_.count(id) > 0) {
        return true;
    }

    if (backend_ == Backend::SentencePiece && processor_) {
        return processor_->IsControl(id) || processor_->IsUnused(id);
    }

    return false;
}

// ---- Common helpers -------------------------------------------------------

std::string Tokenizer::extract_token_string(const json& value) {
    if (value.is_string()) {
        return value.get<std::string>();
    }
    if (value.is_object()) {
        if (value.contains("content") && value["content"].is_string()) {
            return value["content"].get<std::string>();
        }
    }
    return {};
}

int Tokenizer::resolve_token_id(const json& value) const {
    const std::string token = extract_token_string(value);
    if (token.empty()) {
        return -1;
    }
    return token_to_id(token);
}

// ---- SentencePiece backend ------------------------------------------------

bool Tokenizer::load_sentencepiece_model(const fs::path& model_path) {
    processor_ = std::make_shared<sentencepiece::SentencePieceProcessor>();
    const auto status = processor_->Load(model_path.string());
    if (!status.ok()) {
        std::cerr << "[Tokenizer] Failed to load SentencePiece model: "
                  << status.ToString() << '\n';
        processor_.reset();
        return false;
    }

    backend_ = Backend::SentencePiece;
    loaded_ = true;

    bos_id_ = processor_->bos_id();
    eos_id_ = processor_->eos_id();
    pad_id_ = processor_->pad_id();

    infer_special_tokens(model_path);

    std::cout << "[Tokenizer] Loaded SentencePiece model from "
              << model_path << '\n';
    return true;
}

void Tokenizer::infer_special_tokens(const fs::path& model_path) {
    const fs::path base_dir = model_path.parent_path();

    if (bos_id_ < 0) {
        bos_id_ = piece_to_id_guarded(*processor_, "<s>");
    }
    if (eos_id_ < 0) {
        eos_id_ = piece_to_id_guarded(*processor_, "</s>");
    }
    if (pad_id_ < 0) {
        pad_id_ = piece_to_id_guarded(*processor_, "<pad>");
    }

    apply_special_tokens_map(base_dir / "special_tokens_map.json");
    apply_tokenizer_config(base_dir / "tokenizer_config.json");

    if (bos_id_ >= 0) special_token_ids_.insert(bos_id_);
    if (eos_id_ >= 0) special_token_ids_.insert(eos_id_);
    if (pad_id_ >= 0) special_token_ids_.insert(pad_id_);
}

void Tokenizer::apply_special_tokens_map(const fs::path& map_path) {
    if (!fs::exists(map_path)) {
        return;
    }

    std::ifstream in(map_path);
    if (!in) {
        std::cerr << "[Tokenizer] Failed to open " << map_path << '\n';
        return;
    }

    json root;
    try {
        in >> root;
    } catch (const std::exception& e) {
        std::cerr << "[Tokenizer] Failed to parse " << map_path
                  << ": " << e.what() << '\n';
        return;
    }

    if (bos_id_ < 0 && root.contains("bos_token")) {
        bos_id_ = resolve_token_id(root["bos_token"]);
    }
    if (eos_id_ < 0 && root.contains("eos_token")) {
        eos_id_ = resolve_token_id(root["eos_token"]);
    }
    if (pad_id_ < 0 && root.contains("pad_token")) {
        pad_id_ = resolve_token_id(root["pad_token"]);
    }

    if (bos_id_ >= 0) special_token_ids_.insert(bos_id_);
    if (eos_id_ >= 0) special_token_ids_.insert(eos_id_);
    if (pad_id_ >= 0) special_token_ids_.insert(pad_id_);
}

void Tokenizer::apply_tokenizer_config(const fs::path& config_path) {
    if (!fs::exists(config_path)) {
        return;
    }

    std::ifstream in(config_path);
    if (!in) {
        std::cerr << "[Tokenizer] Failed to open " << config_path << '\n';
        return;
    }

    json root;
    try {
        in >> root;
    } catch (const std::exception& e) {
        std::cerr << "[Tokenizer] Failed to parse " << config_path
                  << ": " << e.what() << '\n';
        return;
    }

    if (bos_id_ < 0 && root.contains("bos_token")) {
        bos_id_ = resolve_token_id(root["bos_token"]);
    }
    if (eos_id_ < 0 && root.contains("eos_token")) {
        eos_id_ = resolve_token_id(root["eos_token"]);
    }
    if (pad_id_ < 0 && root.contains("pad_token")) {
        pad_id_ = resolve_token_id(root["pad_token"]);
    }

    if (bos_id_ >= 0) special_token_ids_.insert(bos_id_);
    if (eos_id_ >= 0) special_token_ids_.insert(eos_id_);
    if (pad_id_ >= 0) special_token_ids_.insert(pad_id_);
}

// ---- GPT-BPE backend ------------------------------------------------------

bool Tokenizer::load_bpe_from_json(const fs::path& json_path) {
    std::ifstream in(json_path);
    if (!in) {
        std::cerr << "[Tokenizer] Failed to open tokenizer JSON: "
                  << json_path << '\n';
        return false;
    }

    json tokenizer_json;
    try {
        in >> tokenizer_json;
    } catch (const std::exception& e) {
        std::cerr << "[Tokenizer] Failed to parse tokenizer JSON: "
                  << e.what() << '\n';
        return false;
    }

    const auto model_it = tokenizer_json.find("model");
    if (model_it == tokenizer_json.end()) {
        std::cerr << "[Tokenizer] tokenizer.json missing `model` section.\n";
        return false;
    }
    const json& model = *model_it;
    const auto vocab_it = model.find("vocab");
    const auto merges_it = model.find("merges");
    if (vocab_it == model.end() || merges_it == model.end()) {
        std::cerr << "[Tokenizer] tokenizer.json missing vocab/merges.\n";
        return false;
    }

    vocab_.clear();
    id_to_token_.clear();
    merge_ranks_.clear();
    special_tokens_.clear();
    special_tokens_sorted_.clear();
    special_token_ids_.clear();
    bpe_cache_.clear();

    vocab_.reserve(vocab_it->size());
    for (const auto& item : vocab_it->items()) {
        const std::string& token = item.key();
        const int index = item.value().get<int>();
        vocab_[token] = index;
        if (static_cast<std::size_t>(index) >= id_to_token_.size()) {
            id_to_token_.resize(static_cast<std::size_t>(index) + 1);
        }
        id_to_token_[static_cast<std::size_t>(index)] = token;
    }

    merge_ranks_.reserve(merges_it->size());
    for (std::size_t rank = 0; rank < merges_it->size(); ++rank) {
        const auto& pair = (*merges_it)[rank];
        if (!pair.is_array() || pair.size() != 2) {
            continue;
        }
        const std::string first = pair[0].get<std::string>();
        const std::string second = pair[1].get<std::string>();
        merge_ranks_[make_pair_key(first, second)] = rank;
    }

    try {
        pattern_ = std::regex(
            "'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?\\d+| ?[^\\sA-Za-z\\d]+|\\s+",
            std::regex::icase);
    } catch (const std::exception& e) {
        std::cerr << "[Tokenizer] Failed to compile regex: " << e.what() << '\n';
        return false;
    }

    build_byte_maps();
    load_special_tokens(tokenizer_json);
    load_additional_metadata(json_path.parent_path());

    backend_ = Backend::GPTBPE;
    loaded_ = true;

    std::cout << "[Tokenizer] Loaded GPT-BPE tokenizer from "
              << json_path << '\n';

    return true;
}

void Tokenizer::build_byte_maps() {
    byte_encoder_.clear();
    byte_decoder_.clear();

    std::vector<int> bs;
    for (int i = 33; i <= 126; ++i) bs.push_back(i);
    for (int i = 161; i <= 172; ++i) bs.push_back(i);
    for (int i = 174; i <= 255; ++i) bs.push_back(i);

    std::vector<int> cs = bs;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n);
            ++n;
        }
    }

    for (std::size_t i = 0; i < bs.size(); ++i) {
        const int b = bs[i];
        const char32_t codepoint = static_cast<char32_t>(cs[i]);
        const std::string encoded = utf8_encode(codepoint);
        byte_encoder_[static_cast<uint32_t>(b)] = encoded;
        byte_decoder_[encoded] = static_cast<uint32_t>(b);
    }
}

void Tokenizer::load_special_tokens(const json& tokenizer_json) {
    const auto added_tokens_it = tokenizer_json.find("added_tokens");
    if (added_tokens_it == tokenizer_json.end()) {
        return;
    }

    for (const auto& entry : *added_tokens_it) {
        if (!entry.is_object()) {
            continue;
        }
        const bool is_special = entry.value("special", false);
        if (!is_special) {
            continue;
        }
        const std::string token = entry.value("content", std::string{});
        const int id = entry.value("id", -1);
        if (token.empty() || id < 0) {
            continue;
        }
        special_tokens_[token] = id;
        special_token_ids_.insert(id);
    }

    special_tokens_sorted_.assign(special_tokens_.begin(), special_tokens_.end());
    std::sort(special_tokens_sorted_.begin(), special_tokens_sorted_.end(),
              [](const auto& a, const auto& b) {
                  return a.first.size() > b.first.size();
              });
}

void Tokenizer::load_additional_metadata(const fs::path& base_dir) {
    apply_special_tokens_map(base_dir / "special_tokens_map.json");
    apply_tokenizer_config(base_dir / "tokenizer_config.json");
}

void Tokenizer::encode_segment_bpe(const std::string& text, std::vector<int>& output) const {
    if (text.empty()) {
        return;
    }

    std::sregex_iterator it(text.begin(), text.end(), pattern_);
    std::sregex_iterator end;

    if (it == end) {
        encode_token_bpe(text, output);
        return;
    }

    std::size_t cursor = 0;
    for (; it != end; ++it) {
        const auto& match = *it;
        const std::size_t match_pos = static_cast<std::size_t>(match.position());
        if (match_pos > cursor) {
            encode_token_bpe(text.substr(cursor, match_pos - cursor), output);
        }
        encode_token_bpe(match.str(), output);
        cursor = match_pos + static_cast<std::size_t>(match.length());
    }

    if (cursor < text.size()) {
        encode_token_bpe(text.substr(cursor), output);
    }
}

void Tokenizer::encode_token_bpe(const std::string& token, std::vector<int>& output) const {
    if (token.empty()) {
        return;
    }

    std::string encoded;
    encoded.reserve(token.size());
    for (unsigned char byte : token) {
        encoded += byte_encoder_.at(static_cast<uint32_t>(byte));
    }

    const auto pieces = apply_bpe(encoded);
    for (const auto& piece : pieces) {
        auto it = vocab_.find(piece);
        if (it != vocab_.end()) {
            output.push_back(it->second);
            continue;
        }
        const auto chars = split_utf8(piece);
        for (const auto& ch : chars) {
            auto char_it = vocab_.find(ch);
            if (char_it != vocab_.end()) {
                output.push_back(char_it->second);
            }
        }
    }
}

std::vector<std::string> Tokenizer::apply_bpe(const std::string& token) const {
    auto cache_it = bpe_cache_.find(token);
    if (cache_it != bpe_cache_.end()) {
        return cache_it->second;
    }

    std::vector<std::string> symbols = split_utf8(token);
    if (symbols.size() <= 1) {
        bpe_cache_[token] = symbols;
        return symbols;
    }

    auto get_pairs = [](const std::vector<std::string>& list) {
        std::vector<std::pair<std::string, std::string>> pairs;
        if (list.size() < 2) {
            return pairs;
        }
        pairs.reserve(list.size() - 1);
        for (std::size_t i = 0; i + 1 < list.size(); ++i) {
            pairs.emplace_back(list[i], list[i + 1]);
        }
        return pairs;
    };

    std::vector<std::pair<std::string, std::string>> pairs = get_pairs(symbols);
    while (!pairs.empty()) {
        std::size_t best_rank = std::numeric_limits<std::size_t>::max();
        std::pair<std::string, std::string> best_pair;
        bool found = false;

        for (const auto& pair : pairs) {
            const auto key = make_pair_key(pair.first, pair.second);
            auto rank_it = merge_ranks_.find(key);
            if (rank_it == merge_ranks_.end()) {
                continue;
            }
            if (!found || rank_it->second < best_rank) {
                best_rank = rank_it->second;
                best_pair = pair;
                found = true;
            }
        }

        if (!found) {
            break;
        }

        std::vector<std::string> merged;
        merged.reserve(symbols.size());
        for (std::size_t i = 0; i < symbols.size();) {
            if (i + 1 < symbols.size() &&
                symbols[i] == best_pair.first &&
                symbols[i + 1] == best_pair.second) {
                merged.emplace_back(symbols[i] + symbols[i + 1]);
                i += 2;
            } else {
                merged.emplace_back(symbols[i]);
                ++i;
            }
        }
        symbols.swap(merged);
        if (symbols.size() == 1) {
            break;
        }
        pairs = get_pairs(symbols);
    }

    bpe_cache_[token] = symbols;
    return bpe_cache_[token];
}

std::string Tokenizer::make_pair_key(const std::string& first, const std::string& second) {
    std::string key;
    key.reserve(first.size() + 1 + second.size());
    key.append(first);
    key.push_back(kPairSeparator);
    key.append(second);
    return key;
}
