#pragma once

#include <filesystem>
#include <string>
#include <vector>

class ModelConfig {
public:
    ModelConfig() = default;
    explicit ModelConfig(const std::filesystem::path& json_source);
    explicit ModelConfig(const std::string& json_source)
        : ModelConfig(std::filesystem::path(json_source)) {}

    std::vector<std::string> architectures;
    bool block_auto_adjust_ff_dim = false;
    int block_dim = 0;
    int block_ff_dim = 0;
    double block_ffn_dim_multiplier = 0.0;
    double block_mlp_init_scale = 0.0;
    int block_multiple_of = 1;
    double block_norm_eps = 0.0;
    double block_out_init_scale = 0.0;
    bool block_use_swiglu = false;
    bool block_use_xavier_init = false;
    int bos_token_id = 0;
    int conv_L_cache = 0;
    bool conv_bias = false;
    int conv_dim = 0;
    int conv_dim_out = 0;
    bool conv_use_xavier_init = false;
    int eos_token_id = 0;
    int hidden_size = 0;
    double initializer_range = 0.0;
    int intermediate_size = 0;
    std::vector<std::string> layer_types;
    int max_position_embeddings = 0;
    std::string model_type;
    double norm_eps = 0.0;
    int num_attention_heads = 0;
    int num_heads = 0;
    int num_hidden_layers = 0;
    int num_key_value_heads = 0;
    int pad_token_id = 0;
    double rope_theta = 0.0;
    double theta = 0.0;
    bool tie_embedding = false;
    std::string torch_dtype;
    std::string transformers_version;
    bool use_cache = false;
    bool use_pos_enc = false;
    int vocab_size = 0;

    bool load(const std::string& path);
    bool load(const std::filesystem::path& path);

private:
    std::filesystem::path resolve_config_path(const std::filesystem::path& base) const;
};
