#include "ModelRuntime.hpp"
#include <algorithm>
#include <iomanip>
#include <iostream>

namespace {
void print_vector(const std::vector<int>& values) {
    std::cout << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        std::cout << values[i];
        if (i + 1 < values.size()) {
            std::cout << ", ";
        }
    }
    std::cout << "]";
}

void print_shape(const std::vector<size_t>& shape) {
    std::cout << "(";
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i];
        if (i + 1 < shape.size()) {
            std::cout << " x ";
        }
    }
    std::cout << ")";
}
} // namespace

bool ModelRuntime::initialize(const ModelAssets& assets) {
    if (!config_.load(assets.config.string())) {
        std::cerr << "[Runtime] Failed to load config: " << assets.config.string() << std::endl;
        return false;
    }

    if (!tokenizer_.load(assets.tokenizer.string())) {
        std::cerr << "[Runtime] Failed to load tokenizer: " << assets.tokenizer.string() << std::endl;
        return false;
    }

    if (!loader_.load(assets.weights.string())) {
        std::cerr << "[Runtime] Failed to load weights: " << assets.weights.string() << std::endl;
        return false;
    }

    if (!transformer_.init(config_, loader_)) {
        std::cerr << "[Runtime] Transformer initialization failed.\n";
        return false;
    }

    return true;
}

void ModelRuntime::run_smoke_test() {
    print_config_summary();

    const std::string sample_text = "SentencePiece smoke test";
    const auto token_ids = demo_tokenizer(sample_text);

    demo_weights();

    const auto logits = transformer_.forward(token_ids);
    std::cout << "\n[Transformer]\n";
    summarize_logits(logits);

    const int sampled_token = transformer_.sample(logits);
    std::cout << "  Sampled token: " << sampled_token << std::endl;
}

void ModelRuntime::print_config_summary() const {
    std::cout << "\n[ModelConfig]\n";
    std::cout << "  model_type: " << (config_.model_type.empty() ? "unknown" : config_.model_type) << "\n";
    std::cout << "  hidden_size: " << config_.hidden_size << "\n";
    std::cout << "  num_hidden_layers: " << config_.num_hidden_layers << "\n";
    std::cout << "  num_attention_heads: " << config_.num_attention_heads << "\n";
    std::cout << "  vocab_size: " << config_.vocab_size << "\n";
}

std::vector<int> ModelRuntime::demo_tokenizer(const std::string& sample_text) {
    std::cout << "\n[Tokenizer]\n";
    std::cout << "  Input: \"" << sample_text << "\"\n";
    auto token_ids = tokenizer_.encode(sample_text);
    std::cout << "  Tokens: ";
    print_vector(token_ids);
    std::cout << "\n";
    std::cout << "  Decoded: \"" << tokenizer_.decode(token_ids) << "\"\n";
    return token_ids;
}

void ModelRuntime::demo_weights() {
    std::cout << "\n[Weights]\n";
    loader_.print_all_tensors();

    auto embed_shape = loader_.get_shape("model.embed_tokens.weight");
    std::cout << "  Embed shape: ";
    print_shape(embed_shape);
    std::cout << "\n";

    auto embed_weights = loader_.get("model.embed_tokens.weight");
    auto head_weights = loader_.get("lm_head.weight");
    std::cout << std::fixed << std::setprecision(3);

    if (!embed_weights.empty()) {
        std::cout << "  Embed sample: ";
        const size_t count = std::min<size_t>(embed_weights.size(), 6);
        for (size_t i = 0; i < count; ++i) {
            std::cout << embed_weights[i];
            if (i + 1 < count) {
                std::cout << ", ";
            }
        }
        std::cout << "\n";
    }

    if (!head_weights.empty()) {
        std::cout << "  LM head sample: ";
        const size_t count = std::min<size_t>(head_weights.size(), 6);
        for (size_t i = 0; i < count; ++i) {
            std::cout << head_weights[i];
            if (i + 1 < count) {
                std::cout << ", ";
            }
        }
        std::cout << "\n";
    }

    std::cout << std::defaultfloat;
}

void ModelRuntime::summarize_logits(const std::vector<float>& logits) const {
    if (logits.empty()) {
        std::cout << "  Logits not available.\n";
        return;
    }

    const size_t count = std::min<size_t>(logits.size(), 6);
    std::cout << "  Logits sample: ";
    for (size_t i = 0; i < count; ++i) {
        std::cout << logits[i];
        if (i + 1 < count) {
            std::cout << ", ";
        }
    }
    std::cout << "\n";

    const auto max_it = std::max_element(logits.begin(), logits.end());
    const auto max_index = static_cast<size_t>(std::distance(logits.begin(), max_it));
    std::cout << "  Argmax token: " << max_index << " (logit=" << *max_it << ")\n";
}
