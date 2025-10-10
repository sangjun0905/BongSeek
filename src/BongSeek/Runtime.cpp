#include "BongSeek/Runtime.hpp"

#include "BongSeek/ModelConfig.hpp"

#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string_view>
#include <vector>

namespace fs = std::filesystem;

namespace {

fs::path absolute_path(const fs::path& candidate) {
    std::error_code ec;
    fs::path absolute = fs::absolute(candidate, ec);
    return ec ? candidate : absolute;
}

fs::path detect_repo_root(const std::vector<fs::path>& hints) {
    std::error_code ec;
    for (auto hint : hints) {
        if (hint.empty()) {
            continue;
        }
        hint = absolute_path(hint);
        auto current = hint;
        while (!current.empty()) {
            if (fs::exists(current / "model", ec) && !ec) {
                return current;
            }
            const auto parent = current.parent_path();
            if (parent == current) {
                break;
            }
            current = parent;
        }
    }
    return {};
}

fs::path find_existing(const std::vector<fs::path>& candidates, std::string_view label) {
    std::error_code ec;
    for (const auto& candidate : candidates) {
        if (candidate.empty()) {
            continue;
        }
        const auto absolute = absolute_path(candidate);
        if (fs::exists(absolute, ec) && !ec) {
            return absolute;
        }
    }

    std::ostringstream oss;
    oss << "Failed to locate " << label << ". Checked:";
    for (const auto& candidate : candidates) {
        if (!candidate.empty()) {
            oss << "\n  - " << candidate.string();
        }
    }
    throw std::runtime_error(oss.str());
}

Config load_config(const fs::path& config_path) {
    ModelConfig parsed;
    if (!parsed.load(config_path.string())) {
        throw std::runtime_error("Unable to load config file: " + config_path.string());
    }

    Config config;
    config.block_auto_adjust_ff_dim = parsed.block_auto_adjust_ff_dim;
    config.block_dim = parsed.block_dim;
    config.block_ff_dim = parsed.block_ff_dim;
    config.block_ffn_dim_multiplier = parsed.block_ffn_dim_multiplier;
    config.block_mlp_init_scale = parsed.block_mlp_init_scale;
    config.block_multiple_of = parsed.block_multiple_of;
    config.block_norm_eps = parsed.block_norm_eps;
    config.block_out_init_scale = parsed.block_out_init_scale;
    config.block_use_swiglu = parsed.block_use_swiglu;
    config.block_use_xavier_init = parsed.block_use_xavier_init;
    config.bos_token_id = parsed.bos_token_id;
    config.conv_L_cache = parsed.conv_L_cache;
    config.conv_bias = parsed.conv_bias;
    config.conv_dim = parsed.conv_dim;
    config.conv_dim_out = parsed.conv_dim_out;
    config.conv_use_xavier_init = parsed.conv_use_xavier_init;
    config.eos_token_id = parsed.eos_token_id;
    config.hidden_size = parsed.hidden_size;
    config.initializer_range = parsed.initializer_range;
    config.intermediate_size = parsed.intermediate_size;
    if (!parsed.layer_types.empty()) {
        config.layer_types = parsed.layer_types;
    }
    config.max_position_embeddings = parsed.max_position_embeddings;
    if (!parsed.model_type.empty()) {
        config.model_type = parsed.model_type;
    }
    config.norm_eps = parsed.norm_eps;
    config.num_attention_heads = parsed.num_attention_heads;
    config.num_heads = parsed.num_heads;
    config.num_hidden_layers = parsed.num_hidden_layers > 0
        ? parsed.num_hidden_layers
        : static_cast<int>(config.layer_types.size());
    config.num_key_value_heads = parsed.num_key_value_heads;
    config.pad_token_id = parsed.pad_token_id;
    config.rope_theta = parsed.rope_theta;
    config.theta = parsed.theta;
    config.tie_embedding = parsed.tie_embedding;
    if (!parsed.torch_dtype.empty()) {
        config.torch_dtype = parsed.torch_dtype;
    }
    if (!parsed.transformers_version.empty()) {
        config.transformers_version = parsed.transformers_version;
    }
    config.use_cache = parsed.use_cache;
    config.use_pos_enc = parsed.use_pos_enc;
    config.vocab_size = parsed.vocab_size;
    return config;
}

std::size_t normalise_layers_to_run(const Model& model, std::size_t requested) {
    const std::size_t total = model.layer_count();
    if (requested == 0 || requested > total) {
        return total;
    }
    return requested;
}

} // namespace

namespace bongseek {

RuntimeContext initialize_runtime(const RuntimeOptions& options) {
    std::vector<fs::path> root_hints;
    root_hints.push_back(fs::current_path());
    if (options.executable_path) {
        root_hints.push_back(options.executable_path->parent_path());
        root_hints.push_back(options.executable_path->parent_path().parent_path());
    }

    const fs::path repo_root = detect_repo_root(root_hints);

    std::vector<fs::path> weight_candidates;
    std::vector<fs::path> config_candidates;
    std::vector<fs::path> tokenizer_candidates;

    if (options.weights_path) {
        weight_candidates.push_back(*options.weights_path);
    }
    if (options.config_path) {
        config_candidates.push_back(*options.config_path);
    }
    if (options.tokenizer_path) {
        tokenizer_candidates.push_back(*options.tokenizer_path);
    }

    std::vector<fs::path> search_roots;
    if (!repo_root.empty()) {
        search_roots.push_back(repo_root);
    }
    search_roots.push_back(fs::current_path());
    if (options.executable_path) {
        search_roots.push_back(options.executable_path->parent_path());
    }

    for (const auto& root : search_roots) {
        const fs::path model_dir = root / "model";
        weight_candidates.push_back(model_dir / "model.safetensors");
        config_candidates.push_back(model_dir / "config.json");
        tokenizer_candidates.push_back(model_dir / "tokenizer.model");
        tokenizer_candidates.push_back(model_dir / "spiece.model");
        tokenizer_candidates.push_back(model_dir / "tokenizer.spm");
        tokenizer_candidates.push_back(model_dir / "tokenizer.json");
    }

    const fs::path weights_path = find_existing(weight_candidates, "model weights");
    const fs::path config_path = find_existing(config_candidates, "model config");
    const fs::path tokenizer_path = find_existing(tokenizer_candidates, "tokenizer");

    WeightLoader loader;
    if (!loader.load(weights_path.string())) {
        throw std::runtime_error("Failed to load weights metadata from " + weights_path.string());
    }
    MetadataMap metadata = loader.get_tensor_map();
    if (metadata.empty()) {
        throw std::runtime_error("Weights metadata is empty after loading safetensors file.");
    }

    Config config = load_config(config_path);
    Model model(config);

    std::ifstream weight_stream(weights_path, std::ios::binary);
    if (!weight_stream.is_open()) {
        throw std::runtime_error("Failed to reopen weights file for reading: " + weights_path.string());
    }
    model.load_weights(weight_stream, metadata);

    Tokenizer tokenizer;
    if (!tokenizer.load(tokenizer_path.string())) {
        throw std::runtime_error("Failed to load tokenizer from " + tokenizer_path.string());
    }

    RuntimeContext ctx(std::move(config), std::move(model), std::move(tokenizer));
    ctx.repo_root = repo_root;
    ctx.weights_path = weights_path;
    ctx.config_path = config_path;
    ctx.tokenizer_path = tokenizer_path;
    ctx.layers_to_run = normalise_layers_to_run(ctx.model, options.layers_to_run);
    return ctx;
}

Tensor forward_tokens(RuntimeContext& ctx, const std::vector<int>& token_ids) {
    if (token_ids.empty()) {
        throw std::invalid_argument("forward_tokens requires at least one token");
    }

    nb::Tensor<nb::BFloat16, 3> input(1,
                                      static_cast<std::size_t>(token_ids.size()),
                                      static_cast<std::size_t>(1));
    for (std::size_t i = 0; i < token_ids.size(); ++i) {
        input(0, i, 0) = nb::BFloat16(static_cast<float>(token_ids[i]));
    }

    auto variable = bs::Variable::create(input, "chat_input");
    const std::size_t max_layers = ctx.layers_to_run == 0
        ? std::numeric_limits<std::size_t>::max()
        : ctx.layers_to_run;
    auto output = ctx.model.forward(variable, max_layers);
    return output->data;
}

} // namespace bongseek
