#include "BongSeek/AssetLocator.hpp"
#include <filesystem>

namespace fs = std::filesystem;

namespace {
constexpr int kMaxTraversalDepth = 5;

fs::path resolve_relative(const fs::path& relative) {
    fs::path probe = fs::current_path();
    for (int depth = 0; depth < kMaxTraversalDepth; ++depth) {
        fs::path candidate = probe / relative;
        if (fs::exists(candidate)) {
            return fs::weakly_canonical(candidate);
        }
        if (!probe.has_parent_path()) break;
        probe = probe.parent_path();
    }
    return relative;
}

fs::path prefer_existing(const fs::path& primary, const fs::path& fallback) {
    if (!primary.empty() && fs::exists(primary)) {
        return fs::weakly_canonical(primary);
    }
    return resolve_relative(fallback);
}

} // namespace

bool ModelAssets::all_exist() const {
    return missing().empty();
}

std::vector<std::filesystem::path> ModelAssets::missing() const {
    std::vector<fs::path> absent;
    const auto check = [&](const fs::path& path) {
        if (path.empty() || !fs::exists(path)) {
            absent.push_back(path);
        }
    };
    check(config);
    check(tokenizer);
    check(weights);
    return absent;
}

ModelAssets AssetLocator::locate(const std::filesystem::path& model_dir) const {
    fs::path base = model_dir;
    if (!base.empty() && fs::exists(base)) {
        base = fs::weakly_canonical(base);
    }

    const fs::path default_config    = fs::path("src/model/sample_data/sample_config.json");
    const fs::path default_tokenizer = fs::path("src/model/build/sentencepiece/python/test/test_model.model");
    const fs::path default_weights   = fs::path("src/model/sample_data/sample_weights.safetensors");

    ModelAssets assets;
    assets.config = prefer_existing(base.empty() ? fs::path{} : base / "config.json",
                                    default_config);
    assets.tokenizer = prefer_existing(base.empty() ? fs::path{} : base / "tokenizer.model",
                                       default_tokenizer);
    assets.weights = prefer_existing(base.empty() ? fs::path{} : base / "model.safetensors",
                                     default_weights);
    return assets;
}
