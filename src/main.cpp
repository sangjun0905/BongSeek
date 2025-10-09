#include <iostream>
#include <iomanip>
#include "BongTorch/Core.hpp"
#include "BongTorch/Convld.hpp"

// A simplified tensor print function for 3D tensors
void print_tensor(const Tensor& t) {
    std::cout << "Shape: " << t.shape_string() << std::endl;
    if (t.ndim() == 3) {
        std::cout << "[[[";
        for (size_t j = 0; j < t.getShape()[1]; ++j) {
            for (size_t k = 0; k < t.getShape()[2]; ++k) {
                std::cout << static_cast<float>(t(0, j, k)) << (k == t.getShape()[2] - 1 ? "" : ", ");
            }
            if (j < t.getShape()[1] - 1) std::cout << "],\\n  [";
        }
        std::cout << "]]]" << std::endl;
    }
}


int main() {
    using namespace bs;

    std::cout << std::fixed << std::setprecision(4);

    // --- Conv1d Test ---
    std::cout << "--- Conv1d Test ---" << std::endl;

    // 1. Test Parameters
    std::size_t in_channels = 1;
    std::size_t conv_out_channels = 1;
    std::size_t kernel = 3;
    std::size_t in_proj_out_features = 3; // Must be multiple of out_proj_out_features
    std::size_t out_proj_out_features = 1;
    int stride = 1;
    int padding = 1;
    std::size_t groups = 1;

    // 2. Create Conv1d Layer
    auto conv_layer = std::make_shared<Conv1d>(
        in_channels, conv_out_channels, kernel, 
        in_proj_out_features, out_proj_out_features, 
        stride, padding, groups
    );

    // 3. Initialize Weights
    // conv_weight_ shape: (conv_out, in_channels/groups, kernel) = (1, 1, 3)
    conv_layer->conv_weight()->data.fill(nb::BFloat16(1.0f));

    // in_proj_weight_ shape: (in_proj_out, conv_out, 1) = (3, 1, 1)
    conv_layer->in_proj_weight()->data(0, 0, 0) = nb::BFloat16(1.0f);
    conv_layer->in_proj_weight()->data(1, 0, 0) = nb::BFloat16(2.0f);
    conv_layer->in_proj_weight()->data(2, 0, 0) = nb::BFloat16(3.0f);

    // out_proj_weight_ shape: (out_proj_out, out_proj_input, 1) = (1, 1, 1)
    conv_layer->out_proj_weight()->data.fill(nb::BFloat16(1.0f));

    // 4. Input Tensor
    // Shape: (B=1, C_in=1, S=5)
    TensorShape x_shape = {1, in_channels, 5};
    Tensor x_tensor(x_shape);
    x_tensor(0, 0, 0) = nb::BFloat16(1.0f);
    x_tensor(0, 0, 1) = nb::BFloat16(2.0f);
    x_tensor(0, 0, 2) = nb::BFloat16(3.0f);
    x_tensor(0, 0, 3) = nb::BFloat16(4.0f);
    x_tensor(0, 0, 4) = nb::BFloat16(5.0f);
    auto var_x = Variable::create(x_tensor, "x");

    // 5. Forward Pass
    auto var_y = conv_layer->forward(var_x);

    // 6. Print Results
    std::cout << "Result of Conv1d(x):" << std::endl;
    print_tensor(var_y->data);

    std::cout << "\nExpected result (approximate due to gating function):" << std::endl;
    std::cout << "Shape: (1, 1, 5)" << std::endl;
    std::cout << "[[[12.0000, 24.0000, 36.0000, 48.0000, 36.0000]]]" << std::endl;

    return 0;
}
