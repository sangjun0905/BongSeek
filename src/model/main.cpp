#include <iostream>
#include <filesystem>
#include "AssetLocator.hpp"
#include "ModelRuntime.hpp"

int main(int argc, char** argv) {
    std::cout << "=== LiquidAI Tokenizer & WeightLoader Smoke Test ===\n";

    std::filesystem::path model_dir;
    if (argc > 1) {
        model_dir = std::filesystem::path(argv[1]);
    }

    AssetLocator locator;
    ModelAssets assets = locator.locate(model_dir);

    std::cout << "\n[Asset Paths]\n";
    std::cout << "  Config:    " << assets.config.string() << "\n";
    std::cout << "  Tokenizer: " << assets.tokenizer.string() << "\n";
    std::cout << "  Weights:   " << assets.weights.string() << "\n";

    const auto missing_assets = assets.missing();
    if (!missing_assets.empty()) {
        std::cerr << "\n[Error] Missing asset files:\n";
        for (const auto& path : missing_assets) {
            if (path.empty()) {
                std::cerr << "  - <unspecified>\n";
            } else {
                std::cerr << "  - " << path.string() << "\n";
            }
        }
        return 1;
    }

    ModelRuntime runtime;
    if (!runtime.initialize(assets)) {
        std::cerr << "\n[Error] Failed to initialize model runtime.\n";
        return 1;
    }

    runtime.run_smoke_test();

    std::cout << "\n[Done] Sample assets loaded successfully.\n";
    return 0;
}
