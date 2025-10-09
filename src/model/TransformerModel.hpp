#pragma once

#include <memory>
#include <string>
#include <vector>
#include "LayerWeights.hpp"
#include "ModelConfig.hpp"
#include "WeightLoader.hpp"

class TransformerModel {
public:
    bool init(const ModelConfig& cfg, WeightLoader& loader);
    std::vector<float> forward(const std::vector<int>& input_ids);
    int sample(const std::vector<float>& logits);

private:
    ModelConfig config_;
    std::vector<float> embedding_;
    std::vector<float> embedding_norm_;
    std::vector<float> lm_head_;
    std::vector<std::unique_ptr<BaseLayerWeights>> layers_;
};
