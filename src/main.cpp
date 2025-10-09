// src/main.cpp
#include <algorithm>
#include <iomanip>
#include <iostream>

#include "BongTorch/Core.hpp"
#include "BongTorch/GQAAttention.hpp"

using namespace bs;

// Fill helper used for deterministic tensor/parameter init.
static void fill_tensor(Tensor& tensor, float start, float step) {
    auto* ptr = tensor.data();
    const std::size_t total = tensor.size();
    float value = start;
    for (std::size_t i = 0; i < total; ++i) {
        ptr[i] = static_cast<TensorValueType>(value);
        value += step;
    }
}

static void print_slice(const Tensor& tensor,
                        std::size_t b,
                        std::size_t s,
                        std::size_t count) {
    std::cout << "[";
    for (std::size_t d = 0; d < count; ++d) {
        std::cout << std::fixed << std::setprecision(6)
                  << static_cast<float>(tensor(b, s, d));
        if (d + 1 != count) {
            std::cout << ", ";
        }
    }
    std::cout << "]\n";
}

int main() {
    const std::size_t batch        = 1;
    const std::size_t seq          = 4;
    const std::size_t head_dim     = 8;
    const std::size_t num_heads    = 4;      // model dim = 32
    const std::size_t num_kv_heads = 2;
    const std::size_t model_dim    = head_dim * num_heads;

    Tensor input({batch, seq, model_dim});
    fill_tensor(input, 0.01f, 0.01f);
    auto input_var = Variable::create(input, "input");

    auto attn = std::make_shared<GQAAttention>(
        static_cast<int>(model_dim),
        static_cast<int>(num_heads),
        static_cast<int>(num_kv_heads),
        static_cast<int>(head_dim));

    float scale = 0.05f;
    for (const auto& param : attn->get_parameters()) {
        fill_tensor(param->data, scale, 0.02f);
        scale += 0.05f;
    }

    auto output = attn->forward(input_var);

    std::cout << "Input shape  : " << input.shape_string() << '\n';
    std::cout << "Output shape : " << output->data.shape_string() << '\n';

    std::cout << "\nOutput slice (batch 0, token 0, first 8 dims):\n";
    print_slice(output->data, 0, 0,
                std::min<std::size_t>(8, output->data.getShape()[2]));

    std::cout << "\nGQAAttention demo finished.\n";
    return 0;
}
