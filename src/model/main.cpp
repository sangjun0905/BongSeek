#include <iostream>
#include <filesystem>
#include <vector>
#include <iomanip>
#include "ModelConfig.hpp"
#include "Tokenizer.hpp"
#include "WeightLoader.hpp"
#include "TransformerModel.hpp"

namespace fs = std::filesystem;

static fs::path resolve_path(const fs::path& relative) {
    fs::path probe = fs::current_path();
    for (int i = 0; i < 5; ++i) {
        fs::path candidate = probe / relative;
        if (fs::exists(candidate)) {
            return fs::weakly_canonical(candidate);
        }
        if (!probe.has_parent_path()) break;
        probe = probe.parent_path();
    }
    return relative;
}

static void print_vector(const std::vector<int>& values) {
    std::cout << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        std::cout << values[i];
        if (i + 1 < values.size()) std::cout << ", ";
    }
    std::cout << "]";
}

static void print_shape(const std::vector<size_t>& shape) {
    std::cout << "(";
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i];
        if (i + 1 < shape.size()) std::cout << " x ";
    }
    std::cout << ")";
}

int main() {
    std::cout << "=== LiquidAI Tokenizer & WeightLoader Smoke Test ===\n";

    const fs::path config_path = resolve_path("src/model/sample_data/sample_config.json");
    const fs::path tokenizer_path = resolve_path("src/model/build/sentencepiece/python/test/test_model.model");
    const fs::path weight_path = resolve_path("src/model/sample_data/sample_weights.safetensors");

    std::cout << "\n[Asset Paths]\n";
    std::cout << "  Config:     " << config_path.string() << "\n";
    std::cout << "  Tokenizer:  " << tokenizer_path.string() << "\n";
    std::cout << "  Weights:    " << weight_path.string() << "\n";

    if (!fs::exists(config_path) || !fs::exists(tokenizer_path) || !fs::exists(weight_path)) {
        std::cerr << "[Error] One or more asset files are missing.\n";
        return 1;
    }

    ModelConfig config{};
    if (!config.load(config_path.string())) {
        std::cerr << "[Error] Failed to load sample config.\n";
        return 1;
    }
    std::cout << "\n[ModelConfig]\n";
    std::cout << "  vocab_size=" << config.vocab_size
              << " hidden_size=" << config.hidden_size
              << " layers=" << config.num_hidden_layers << "\n";

    Tokenizer tokenizer;
    if (!tokenizer.load(tokenizer_path.string())) {
        std::cerr << "[Error] Tokenizer model load failed.\n";
        return 1;
    }

    const std::string sample_text = "SentencePiece smoke test";
    auto token_ids = tokenizer.encode(sample_text);
    std::cout << "\n[Tokenizer]\n";
    std::cout << "  Input:  \"" << sample_text << "\"\n";
    std::cout << "  Tokens: ";
    print_vector(token_ids);
    std::cout << "\n";
    std::cout << "  Decoded: \"" << tokenizer.decode(token_ids) << "\"\n";

    WeightLoader loader;
    if (!loader.load(weight_path.string())) {
        std::cerr << "[Error] Weight file load failed.\n";
        return 1;
    }

    std::cout << "\n[Weights]\n";
    loader.print_all_tensors();

    auto embed_shape = loader.get_shape("model.embed_tokens.weight");
    std::cout << "  Embed shape: ";
    print_shape(embed_shape);
    std::cout << "\n";

    auto embed_weights = loader.get("model.embed_tokens.weight");
    auto head_weights = loader.get("lm_head.weight");
    std::cout << std::fixed << std::setprecision(3);
    if (!embed_weights.empty()) {
        std::cout << "  Embed sample: ";
        const size_t count = std::min<size_t>(embed_weights.size(), 6);
        for (size_t i = 0; i < count; ++i) {
            std::cout << embed_weights[i];
            if (i + 1 < count) std::cout << ", ";
        }
        std::cout << "\n";
    }
    if (!head_weights.empty()) {
        std::cout << "  LM head sample: ";
        const size_t count = std::min<size_t>(head_weights.size(), 6);
        for (size_t i = 0; i < count; ++i) {
            std::cout << head_weights[i];
            if (i + 1 < count) std::cout << ", ";
        }
        std::cout << "\n";
    }

    TransformerModel transformer;
    transformer.init(config, loader);

    std::cout << "\n[Done] Sample assets loaded successfully.\n";
    return 0;
}
