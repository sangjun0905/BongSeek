#pragma once
#include <string>
#include <vector>
#include "WeightLoader.hpp"

class LiquidBlock {
private:
    int layer_idx;

public:
    std::vector<float> conv_in_proj;
    std::vector<float> conv_kernel;
    std::vector<float> conv_out_proj;

    std::vector<float> ffn_w1;
    std::vector<float> ffn_w2;
    std::vector<float> ffn_w3;

    std::vector<float> op_norm;
    std::vector<float> ffn_norm;

    void init(int idx, WeightLoader& loader);
};
