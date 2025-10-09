#pragma once

#include <filesystem>
#include <vector>

struct ModelAssets {
    std::filesystem::path config;
    std::filesystem::path tokenizer;
    std::filesystem::path weights;

    [[nodiscard]] bool all_exist() const;
    [[nodiscard]] std::vector<std::filesystem::path> missing() const;
};

class AssetLocator {
public:
    ModelAssets locate(const std::filesystem::path& model_dir = {}) const;
};
