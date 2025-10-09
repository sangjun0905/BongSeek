#include "LayerWeights.hpp"
#include <sstream>
#include <stdexcept>

namespace {
std::string layer_prefix(int index) {
    std::ostringstream oss;
    oss << "model.layers." << index;
    return oss.str();
}
} // namespace

BaseLayerWeights::BaseLayerWeights(int index, Kind kind)
    : index_(index), kind_(kind) {}

std::vector<float> BaseLayerWeights::fetch(WeightLoader& loader, const std::string& tensor_name) {
    auto tensor = loader.get(tensor_name);
    if (tensor.empty()) {
        std::ostringstream oss;
        oss << "[LayerWeights] Missing tensor: " << tensor_name;
        throw std::runtime_error(oss.str());
    }
    return tensor;
}

void BaseLayerWeights::load_common(WeightLoader& loader, const std::string& layer_prefix) {
    const std::string ffn_prefix = layer_prefix + ".feed_forward";
    ffn_.w1 = fetch(loader, ffn_prefix + ".w1.weight");
    ffn_.w2 = fetch(loader, ffn_prefix + ".w2.weight");
    ffn_.w3 = fetch(loader, ffn_prefix + ".w3.weight");

    norms_.operator_norm = fetch(loader, layer_prefix + ".operator_norm.weight");
    norms_.ffn_norm = fetch(loader, layer_prefix + ".ffn_norm.weight");
}

ConvLayerWeights::ConvLayerWeights(int index)
    : BaseLayerWeights(index, Kind::Conv) {}

void ConvLayerWeights::load(WeightLoader& loader) {
    const std::string prefix = layer_prefix(index());
    load_common(loader, prefix);

    const std::string conv_prefix = prefix + ".conv";
    conv_in_proj_ = fetch(loader, conv_prefix + ".in_proj.weight");
    conv_kernel_ = fetch(loader, conv_prefix + ".conv.weight");
    conv_out_proj_ = fetch(loader, conv_prefix + ".out_proj.weight");
}

std::string ConvLayerWeights::debug_label() const {
    std::ostringstream oss;
    oss << "ConvLayer(index=" << index() << ")";
    return oss.str();
}

AttentionLayerWeights::AttentionLayerWeights(int index)
    : BaseLayerWeights(index, Kind::Attention) {}

void AttentionLayerWeights::load(WeightLoader& loader) {
    const std::string prefix = layer_prefix(index());
    load_common(loader, prefix);

    const std::string attn_prefix = prefix + ".self_attn";
    q_layernorm_ = fetch(loader, attn_prefix + ".q_layernorm.weight");
    k_layernorm_ = fetch(loader, attn_prefix + ".k_layernorm.weight");

    q_proj_ = fetch(loader, attn_prefix + ".q_proj.weight");
    k_proj_ = fetch(loader, attn_prefix + ".k_proj.weight");
    v_proj_ = fetch(loader, attn_prefix + ".v_proj.weight");
    out_proj_ = fetch(loader, attn_prefix + ".out_proj.weight");
}

std::string AttentionLayerWeights::debug_label() const {
    std::ostringstream oss;
    oss << "AttentionLayer(index=" << index() << ")";
    return oss.str();
}
