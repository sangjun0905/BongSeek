#pragma once

#include <memory>
#include <string>
#include <vector>

#include "ModelConfig.hpp"
#include "WeightLoader.hpp"

class BaseLayerWeights {
public:
    explicit BaseLayerWeights(int index) : index_(index) {}
    virtual ~BaseLayerWeights() = default;

    virtual void load(WeightLoader& loader) = 0;
    [[nodiscard]] virtual std::string debug_label() const = 0;

protected:
    int index_;
};

class ConvLayerWeights : public BaseLayerWeights {
public:
    using BaseLayerWeights::BaseLayerWeights;

    void load(WeightLoader&) override {}
    [[nodiscard]] std::string debug_label() const override {
        return "ConvLayer[" + std::to_string(index_) + "]";
    }
};

class AttentionLayerWeights : public BaseLayerWeights {
public:
    using BaseLayerWeights::BaseLayerWeights;

    void load(WeightLoader&) override {}
    [[nodiscard]] std::string debug_label() const override {
        return "AttentionLayer[" + std::to_string(index_) + "]";
    }
};

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
