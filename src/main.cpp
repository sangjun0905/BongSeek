#include <iostream>
#include <iomanip>

#include <cmath>
#include "BongTorch/Core.hpp"
#include "BongTorch/RoPE.hpp"

void print_tensor(const Tensor& t) {
    std::cout << "Shape: " << t.shape_string() << std::endl;
    if (t.ndim() == 3) {
        std::cout << "[[[";
        for (size_t j = 0; j < t.getShape()[1]; ++j) {
            for (size_t k = 0; k < t.getShape()[2]; ++k) {
                std::cout << static_cast<float>(t(0, j, k)) << (k == t.getShape()[2] - 1 ? "" : ", ");
            }
            if (j < t.getShape()[1] - 1) std::cout << "],\n  [";
        }
        std::cout << "]]]" << std::endl;
    }
}

int main() {
    using namespace bs;

    std::cout << std::fixed << std::setprecision(4);

    // --- RoPE Test ---
    std::cout << "--- RoPE Test ---" << std::endl;

    // 1. Test Parameters
    const int seq_len = 2;
    const int dim = 4;
    const int d_half = dim / 2;

    // 2. Input Tensor x
    Tensor x_tensor({1, seq_len, dim});
    x_tensor(0, 0, 0) = nb::BFloat16(1.0f); x_tensor(0, 0, 1) = nb::BFloat16(2.0f); x_tensor(0, 0, 2) = nb::BFloat16(3.0f); x_tensor(0, 0, 3) = nb::BFloat16(4.0f);
    x_tensor(0, 1, 0) = nb::BFloat16(5.0f); x_tensor(0, 1, 1) = nb::BFloat16(6.0f); x_tensor(0, 1, 2) = nb::BFloat16(7.0f); x_tensor(0, 1, 3) = nb::BFloat16(8.0f);
    auto var_x = Variable::create(x_tensor, "x");

    // 3. Create C and S tensors for positional encoding
    Tensor C_tensor({1, seq_len, d_half});
    Tensor S_tensor({1, seq_len, d_half});

    double inv_base = 1.0 / 10000.0;
    for (int m = 0; m < seq_len; ++m) {
        for (int i = 0; i < d_half; ++i) {
            double theta = std::pow(inv_base, static_cast<double>(2 * i) / dim);
            double m_theta = static_cast<double>(m) * theta;
            C_tensor(0, m, i) = nb::BFloat16(static_cast<float>(std::cos(m_theta)));
            S_tensor(0, m, i) = nb::BFloat16(static_cast<float>(std::sin(m_theta)));
        }
    }
    auto var_C = Variable::create(C_tensor, "C");
    auto var_S = Variable::create(S_tensor, "S");

    // 4. Forward Pass
    auto var_y = rope(var_x, var_C, var_S);

    // 5. Print Results
    std::cout << "Result of RoPE(x):" << std::endl;
    print_tensor(var_y->data);

    std::cout << "\nExpected result (approximate):" << std::endl;
    std::cout << "Shape: (1, 2, 4)" << std::endl;
    std::cout << "[[[1.0000, 2.0000, 3.0000, 4.0000],\n  [-3.1890, 5.9198, 7.9896, 8.0595]]]" << std::endl;

    return 0;
}
