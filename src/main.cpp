#include <iostream>
#include "BongTorch/Core.hpp"
#include "BongTorch/Conv1d.hpp"

void test_conv1d_layer() {
    std::cout << "\n--- Testing Conv1DLayer ---" << std::endl;

    using namespace bs;

    int batch_size = 1;
    int in_channels = 3;
    int out_channels = 5;
    int seq_length = 10;
    int kernel_size = 3;
    int stride = 1;
    int padding = 1;

    auto conv_layer = std::make_shared<Conv1DLayer>(in_channels, out_channels, kernel_size, stride, padding);

    std::array<size_t, 3> shape = {(size_t)batch_size, (size_t)in_channels, (size_t)seq_length};
    auto input_data = nb::randn(shape);
    auto x = Variable::create(input_data, "input");

    std::cout << "Input shape: (" << x->shape()[0] << ", " << x->shape()[1] << ", " << x->shape()[2] << ")" << std::endl;

    auto output = conv_layer->forward(x);

    std::cout << "Output shape: (" << output->shape()[0] << ", " << output->shape()[1] << ", " << output->shape()[2] << ")" << std::endl;
    
    std::cout << "Expected output shape: (" << batch_size << ", " << out_channels << ", " << seq_length << ")" << std::endl;

    std::cout << "Conv1DLayer test finished." << std::endl;
}

int main() {
    std::cout << "Running BongTorch Demo..." << std::endl;
    bs::demo();
    std::cout << "BongTorch Demo finished." << std::endl;

    test_conv1d_layer();

    return 0;
}