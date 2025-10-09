#include <iostream>
#include <iomanip>
#include "BongTorch/Core.hpp"
#include "BongTorch/RMSNorm.hpp"

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

    // Test RMSNorm
    std::cout << "--- RMSNorm Test ---" << std::endl;

    int dim = 4;
    auto rms_norm = std::make_shared<RMSNorm>(dim);

    TensorShape shape_x = {1, 1, static_cast<size_t>(dim)};
    Tensor x_tensor(shape_x);
    x_tensor(0, 0, 0) = nb::BFloat16(1.0f);
    x_tensor(0, 0, 1) = nb::BFloat16(2.0f);
    x_tensor(0, 0, 2) = nb::BFloat16(3.0f);
    x_tensor(0, 0, 3) = nb::BFloat16(4.0f);

    auto var_x = Variable::create(x_tensor, "x");

    auto var_y = rms_norm->forward(var_x);

    std::cout << "Result of RMSNorm(x):" << std::endl;
    print_tensor(var_y->data);

    std::cout << "\nExpected result:" << std::endl;
    std::cout << "Shape: (1, 1, 4)" << std::endl;
    std::cout << "[[[0.3651, 0.7303, 1.0954, 1.4606]]]" << std::endl;

    return 0;
}
