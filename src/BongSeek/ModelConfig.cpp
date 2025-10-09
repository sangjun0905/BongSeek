#include "BongSeek/ModelConfig.hpp"
#include <exception>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace {
std::vector<std::string> read_string_array(const json& data, const char* key) {
    std::vector<std::string> out;
    if (!data.contains(key)) return out;
    const auto& node = data.at(key);
    if (!node.is_array()) return out;
    for (const auto& item : node) {
        if (item.is_string()) out.push_back(item.get<std::string>());
    }
    return out;
}
} // namespace

bool ModelConfig::load(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "[ModelConfig] 파일 읽기 실패: " << path << std::endl;
        return false;
    }

    json j;
    try {
        file >> j;
    } catch (const std::exception& e) {
        std::cerr << "[ModelConfig] JSON 파싱 실패: " << e.what() << std::endl;
        return false;
    }

    *this = ModelConfig();

    architectures = read_string_array(j, "architectures");
    layer_types = read_string_array(j, "layer_types");

    block_auto_adjust_ff_dim = j.value("block_auto_adjust_ff_dim", block_auto_adjust_ff_dim);
    block_dim = j.value("block_dim", block_dim);
    block_ff_dim = j.value("block_ff_dim", block_ff_dim);
    block_ffn_dim_multiplier = j.value("block_ffn_dim_multiplier", block_ffn_dim_multiplier);
    block_mlp_init_scale = j.value("block_mlp_init_scale", block_mlp_init_scale);
    block_multiple_of = j.value("block_multiple_of", block_multiple_of);
    block_norm_eps = j.value("block_norm_eps", block_norm_eps);
    block_out_init_scale = j.value("block_out_init_scale", block_out_init_scale);
    block_use_swiglu = j.value("block_use_swiglu", block_use_swiglu);
    block_use_xavier_init = j.value("block_use_xavier_init", block_use_xavier_init);
    bos_token_id = j.value("bos_token_id", bos_token_id);
    conv_L_cache = j.value("conv_L_cache", conv_L_cache);
    conv_bias = j.value("conv_bias", conv_bias);
    conv_dim = j.value("conv_dim", conv_dim);
    conv_dim_out = j.value("conv_dim_out", conv_dim_out);
    conv_use_xavier_init = j.value("conv_use_xavier_init", conv_use_xavier_init);
    eos_token_id = j.value("eos_token_id", eos_token_id);
    hidden_size = j.value("hidden_size", hidden_size);
    initializer_range = j.value("initializer_range", initializer_range);
    intermediate_size = j.value("intermediate_size", intermediate_size);
    max_position_embeddings = j.value("max_position_embeddings", max_position_embeddings);
    model_type = j.value("model_type", model_type);
    norm_eps = j.value("norm_eps", norm_eps);
    num_attention_heads = j.value("num_attention_heads", num_attention_heads);
    num_heads = j.value("num_heads", num_heads);
    num_hidden_layers = j.value("num_hidden_layers", num_hidden_layers);
    num_key_value_heads = j.value("num_key_value_heads", num_key_value_heads);
    pad_token_id = j.value("pad_token_id", pad_token_id);
    rope_theta = j.value("rope_theta", rope_theta);
    theta = j.value("theta", theta);
    tie_embedding = j.value("tie_embedding", tie_embedding);
    torch_dtype = j.value("torch_dtype", torch_dtype);
    transformers_version = j.value("transformers_version", transformers_version);
    use_cache = j.value("use_cache", use_cache);
    use_pos_enc = j.value("use_pos_enc", use_pos_enc);
    vocab_size = j.value("vocab_size", vocab_size);

    if (num_hidden_layers == 0 && !layer_types.empty()) {
        num_hidden_layers = static_cast<int>(layer_types.size());
    }

    std::cout << "[ModelConfig] 로드 완료: model_type="
              << (model_type.empty() ? "unknown" : model_type)
              << ", layers=" << num_hidden_layers
              << ", hidden_size=" << hidden_size
              << ", vocab_size=" << vocab_size << std::endl;
    return true;
}
