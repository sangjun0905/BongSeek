#include "LiquidBlock.hpp"
#include <sstream>
#include <iostream>

void LiquidBlock::init(int idx, WeightLoader& loader) {
    layer_idx = idx;
    std::stringstream prefix;
    prefix << "model.layers." << idx;

    auto safe_get = [&](const std::string& name) -> std::vector<float> {
        auto data = loader.get(name);
        if (data.empty()) std::cerr << "[LiquidBlock] Missing: " << name << std::endl;
        return data;
    };

    conv_in_proj = safe_get(prefix.str() + ".conv.in_proj.weight");
    conv_kernel  = safe_get(prefix.str() + ".conv.conv.weight");
    conv_out_proj= safe_get(prefix.str() + ".conv.out_proj.weight");

    ffn_w1 = safe_get(prefix.str() + ".feed_forward.w1.weight");
    ffn_w2 = safe_get(prefix.str() + ".feed_forward.w2.weight");
    ffn_w3 = safe_get(prefix.str() + ".feed_forward.w3.weight");

    op_norm  = safe_get(prefix.str() + ".operator_norm.weight");
    ffn_norm = safe_get(prefix.str() + ".ffn_norm.weight");

    std::cout << "[LiquidBlock] Layer " << idx << " 로드 완료." << std::endl;
}
