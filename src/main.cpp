#include <iostream>
#include <iomanip>
#include "BongTorch/Core.hpp"
#include "BongTorch/FFN_SWiGLU.hpp"

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

    std::cout << std::fixed << std::setprecision(4);

    // Test FFN_SWiGLU
    std::cout << "--- FFN_SWiGLU Test ---" << std::endl;

    int embed_dim = 2;
    int hidden_dim = 3;

    auto ffn = std::make_shared<FFN_SWiGLU>(embed_dim, hidden_dim);

    // Initialize weights for testing
    auto& W1 = ffn->get_gate_linear()->weight()->data;
    W1(0, 0, 0) = nb::BFloat16(1.0f); W1(0, 0, 1) = nb::BFloat16(0.0f);
    W1(0, 1, 0) = nb::BFloat16(0.0f); W1(0, 1, 1) = nb::BFloat16(1.0f);
    W1(0, 2, 0) = nb::BFloat16(1.0f); W1(0, 2, 1) = nb::BFloat16(1.0f);

    auto& W2 = ffn->get_value_linear()->weight()->data;
    W2(0, 0, 0) = nb::BFloat16(1.0f); W2(0, 0, 1) = nb::BFloat16(0.0f);
    W2(0, 1, 0) = nb::BFloat16(1.0f); W2(0, 1, 1) = nb::BFloat16(0.0f);
    W2(0, 2, 0) = nb::BFloat16(1.0f); W2(0, 2, 1) = nb::BFloat16(0.0f);

    auto& W3 = ffn->get_down_linear()->weight()->data;
    W3(0, 0, 0) = nb::BFloat16(1.0f); W3(0, 0, 1) = nb::BFloat16(0.0f); W3(0, 0, 2) = nb::BFloat16(0.0f);
    W3(0, 1, 0) = nb::BFloat16(0.0f); W3(0, 1, 1) = nb::BFloat16(1.0f); W3(0, 1, 2) = nb::BFloat16(1.0f);

    // Input tensor
    TensorShape shape_x = {1, 1, 2};
    Tensor x_tensor(shape_x);
    x_tensor(0, 0, 0) = nb::BFloat16(1.0f);
    x_tensor(0, 0, 1) = nb::BFloat16(2.0f);
    auto var_x = Variable::create(x_tensor, "x");

    // Forward pass
    auto var_y = ffn->forward(var_x);

    std::cout << "Result of FFN_SWiGLU(x):" << std::endl;
    print_tensor(var_y->data);

    std::cout << "\nExpected result (approximate due to BFloat16 precision):" << std::endl;
    std::cout << "Shape: (1, 1, 2)" << std::endl;
    std::cout << "[[[0.7311, 4.6193]]]" << std::endl;

    return 0;
}