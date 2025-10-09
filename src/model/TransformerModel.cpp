#include "TransformerModel.hpp"
#include <algorithm>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>

namespace {
std::vector<float> fetch_required(WeightLoader& loader, const std::string& tensor_name) {
    auto tensor = loader.get(tensor_name);
    if (tensor.empty()) {
        std::ostringstream oss;
        oss << "[TransformerModel] Missing tensor: " << tensor_name;
        throw std::runtime_error(oss.str());
    }
    return tensor;
}

std::unique_ptr<BaseLayerWeights> make_layer(int index, const std::string& type) {
    if (type == "conv") {
        return std::make_unique<ConvLayerWeights>(index);
    }
    if (type == "full_attention") {
        return std::make_unique<AttentionLayerWeights>(index);
    }

    std::ostringstream oss;
    oss << "[TransformerModel] Unsupported layer type '" << type
        << "' at index " << index;
    throw std::runtime_error(oss.str());
}
} // namespace

bool TransformerModel::init(const ModelConfig& cfg, WeightLoader& loader) {
    config_ = cfg;
    layers_.clear();

    try {
        embedding_ = fetch_required(loader, "model.embed_tokens.weight");

        try {
            embedding_norm_ = fetch_required(loader, "model.embedding_norm.weight");
        } catch (const std::exception&) {
            embedding_norm_.clear();
            std::cout << "[TransformerModel] embedding_norm.weight not found; continuing without it.\n";
        }

        lm_head_ = fetch_required(loader, "lm_head.weight");

        const auto& layer_types = config_.layer_types;
        if (layer_types.size() < static_cast<size_t>(config_.num_hidden_layers)) {
            std::ostringstream oss;
            oss << "[TransformerModel] layer_types size (" << layer_types.size()
                << ") is smaller than num_hidden_layers (" << config_.num_hidden_layers << ")";
            throw std::runtime_error(oss.str());
        }

        const bool has_layer_weights = loader.has("model.layers.0.feed_forward.w1.weight");
        if (!has_layer_weights) {
            std::cout << "[TransformerModel] Layer weights not found in safetensors; skipping block loading.\n";
        } else {
            for (int i = 0; i < config_.num_hidden_layers; ++i) {
                auto layer = make_layer(i, layer_types.at(static_cast<size_t>(i)));
                layer->load(loader);
                std::cout << "  • " << layer->debug_label() << " loaded\n";
                layers_.push_back(std::move(layer));
            }
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        layers_.clear();
        return false;
    }

    std::cout << "[TransformerModel] Loaded " << layers_.size()
              << " layers (hidden_size=" << config_.hidden_size << ")\n";
    return true;
}

std::vector<float> TransformerModel::forward(const std::vector<int>& input_ids) {
    if (config_.vocab_size <= 0) return {};

    // Placeholder implementation: the actual decoder stack is not yet implemented.
    std::vector<float> logits(static_cast<size_t>(config_.vocab_size), 0.0f);
    if (!lm_head_.empty()) {
        const size_t copy_count = std::min(logits.size(), lm_head_.size());
        std::copy_n(lm_head_.begin(), copy_count, logits.begin());
    }

    if (!input_ids.empty()) {
        const size_t idx = static_cast<size_t>(input_ids.back()) % logits.size();
        logits[idx] += 1.0f;
    }

    return logits;
}

int TransformerModel::sample(const std::vector<float>& logits) {
    if (logits.empty()) return 0;
    const auto max_it = std::max_element(logits.begin(), logits.end());
    return static_cast<int>(std::distance(logits.begin(), max_it));
}
