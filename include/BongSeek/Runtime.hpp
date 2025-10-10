#pragma once

#include "Config.hpp"
#include "Model.hpp"
#include "Tokenizer.hpp"
#include "WeightLoader.hpp"
#include "BongTorch/Core.hpp"
#include "NumBong/Tensor.hpp"

#include <filesystem>
#include <optional>
#include <string>
#include <vector>
#include <utility>

namespace bongseek {

struct RuntimeOptions {
    std::optional<std::filesystem::path> executable_path;
    std::optional<std::filesystem::path> weights_path;
    std::optional<std::filesystem::path> config_path;
    std::optional<std::filesystem::path> tokenizer_path;
    std::size_t layers_to_run = 0; // 0 => run all layers
};

struct RuntimeContext {
    RuntimeContext(Config cfg, Model mdl, Tokenizer tok)
        : config(std::move(cfg)),
          model(std::move(mdl)),
          tokenizer(std::move(tok)) {}

    RuntimeContext(const RuntimeContext&) = delete;
    RuntimeContext& operator=(const RuntimeContext&) = delete;
    RuntimeContext(RuntimeContext&&) noexcept = default;
    RuntimeContext& operator=(RuntimeContext&&) noexcept = default;

    Config config;
    Model model;
    Tokenizer tokenizer;
    std::filesystem::path repo_root;
    std::filesystem::path weights_path;
    std::filesystem::path config_path;
    std::filesystem::path tokenizer_path;
    std::size_t layers_to_run = 0;
};

RuntimeContext initialize_runtime(const RuntimeOptions& options);

Tensor forward_tokens(RuntimeContext& ctx, const std::vector<int>& token_ids);

} // namespace bongseek
