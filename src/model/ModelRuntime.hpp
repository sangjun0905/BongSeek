#pragma once

#include <string>
#include <vector>
#include "AssetLocator.hpp"
#include "ModelConfig.hpp"
#include "Tokenizer.hpp"
#include "TransformerModel.hpp"
#include "WeightLoader.hpp"

class ModelRuntime {
public:
    bool initialize(const ModelAssets& assets);
    void run_smoke_test();

private:
    void print_config_summary() const;
    std::vector<int> demo_tokenizer(const std::string& sample_text);
    void demo_weights();
    void summarize_logits(const std::vector<float>& logits) const;

    ModelConfig config_;
    Tokenizer tokenizer_;
    WeightLoader loader_;
    TransformerModel transformer_;
};
