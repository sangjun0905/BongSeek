#include <iostream>
#include "BongTorch/Core.hpp"
#include "BongTorch/Linear.hpp"

void print_tensor(const Tensor& t) {
    std::cout << "Shape: " << t.shape_string() << std::endl;
    if (t.ndim() == 3) {
        for (size_t i = 0; i < t.getShape()[0]; ++i) {
            std::cout << "[[";
            for (size_t j = 0; j < t.getShape()[1]; ++j) {
                std::cout << "[";
                for (size_t k = 0; k < t.getShape()[2]; ++k) {
                    std::cout << static_cast<float>(t(i, j, k)) << (k == t.getShape()[2] - 1 ? "" : ", ");
                }
                std::cout << "]" << (j == t.getShape()[1] - 1 ? "" : ",\n  ");
            }
            std::cout << "]]" << std::endl;
        }
    }
}

int main() {
    using namespace bs;

    // Test Linear
    std::cout << "--- Linear Test ---" << std::endl;

    int in_features = 3;
    int out_features = 4;

    auto linear = std::make_shared<Linear>(in_features, out_features, false);

    // Initialize weights for testing
    auto& W_tensor = linear->weight()->data;
    for (size_t i = 0; i < out_features; ++i) {
        for (size_t j = 0; j < in_features; ++j) {
            W_tensor(0, i, j) = nb::BFloat16(static_cast<float>(i + 1));
        }
    }

    TensorShape shape_x = {1, 2, 3};
    Tensor x_tensor(shape_x);
    x_tensor(0, 0, 0) = nb::BFloat16(1.0f);
    x_tensor(0, 0, 1) = nb::BFloat16(2.0f);
    x_tensor(0, 0, 2) = nb::BFloat16(3.0f);
    x_tensor(0, 1, 0) = nb::BFloat16(4.0f);
    x_tensor(0, 1, 1) = nb::BFloat16(5.0f);
    x_tensor(0, 1, 2) = nb::BFloat16(6.0f);

    auto var_x = Variable::create(x_tensor, "x");

    auto var_y = linear->forward(var_x);

    std::cout << "Result of Linear(x):" << std::endl;
    print_tensor(var_y->data);

    std::cout << "\nExpected result:" << std::endl;
    std::cout << "Shape: (1, 2, 4)" << std::endl;
    std::cout << "[[[6, 12, 18, 24],\n  [15, 30, 45, 60]]]";

    return 0;
}