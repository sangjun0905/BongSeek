#pragma once

#include <memory>
#include <string>
#include <vector>
#include "WeightLoader.hpp"

struct FeedForwardWeights {
    std::vector<float> w1;
    std::vector<float> w2;
    std::vector<float> w3;
};

struct LayerNormWeights {
    std::vector<float> operator_norm;
    std::vector<float> ffn_norm;
};

class BaseLayerWeights {
public:
    enum class Kind { Conv, Attention };

    BaseLayerWeights(int index, Kind kind);
    virtual ~BaseLayerWeights() = default;

    int index() const { return index_; }
    Kind kind() const { return kind_; }

    const FeedForwardWeights& feed_forward() const { return ffn_; }
    const LayerNormWeights& norms() const { return norms_; }

    virtual void load(WeightLoader& loader) = 0;
    virtual std::string debug_label() const = 0;

protected:
    std::vector<float> fetch(WeightLoader& loader, const std::string& tensor_name);
    void load_common(WeightLoader& loader, const std::string& layer_prefix);

private:
    int index_;
    Kind kind_;
    FeedForwardWeights ffn_;
    LayerNormWeights norms_;
};

class ConvLayerWeights final : public BaseLayerWeights {
public:
    explicit ConvLayerWeights(int index);

    void load(WeightLoader& loader) override;
    std::string debug_label() const override;

    const std::vector<float>& in_proj() const { return conv_in_proj_; }
    const std::vector<float>& kernel() const { return conv_kernel_; }
    const std::vector<float>& out_proj() const { return conv_out_proj_; }

private:
    std::vector<float> conv_in_proj_;
    std::vector<float> conv_kernel_;
    std::vector<float> conv_out_proj_;
};

class AttentionLayerWeights final : public BaseLayerWeights {
public:
    explicit AttentionLayerWeights(int index);

    void load(WeightLoader& loader) override;
    std::string debug_label() const override;

    const std::vector<float>& q_layernorm() const { return q_layernorm_; }
    const std::vector<float>& k_layernorm() const { return k_layernorm_; }
    const std::vector<float>& q_proj() const { return q_proj_; }
    const std::vector<float>& k_proj() const { return k_proj_; }
    const std::vector<float>& v_proj() const { return v_proj_; }
    const std::vector<float>& out_proj() const { return out_proj_; }

private:
    std::vector<float> q_layernorm_;
    std::vector<float> k_layernorm_;
    std::vector<float> q_proj_;
    std::vector<float> k_proj_;
    std::vector<float> v_proj_;
    std::vector<float> out_proj_;
};
