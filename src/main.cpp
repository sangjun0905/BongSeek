#include <iostream>
#include "BongTorch/Core.hpp"
#include "BongTorch/Embedding.hpp"

int main() {
    using namespace bs;

    std::size_t batch = 2;
    std::size_t seq = 3;
    std::size_t vocab = 5;
    std::size_t dim = 4;

    auto model = std::make_shared<Embedding>(vocab, dim);

    TensorShape idx_shape = {batch, seq, 1};
    Tensor idx_data(idx_shape);
    idx_data.fill(nb::BFloat16(0.0f));

    idx_data(0, 0, 0) = nb::BFloat16(1.0f);
    idx_data(0, 1, 0) = nb::BFloat16(3.0f);
    idx_data(0, 2, 0) = nb::BFloat16(2.0f);
    idx_data(1, 0, 0) = nb::BFloat16(4.0f);
    idx_data(1, 1, 0) = nb::BFloat16(0.0f);
    idx_data(1, 2, 0) = nb::BFloat16(1.0f);

    auto x = Variable::create(idx_data, "token_indices");

    auto y = (*model)(x);

    std::cout << "Embedding output shape: "
              << y->data.shape_string() << "\n";
    for (std::size_t b = 0; b < batch; ++b) {
        for (std::size_t s = 0; s < seq; ++s) {
            std::cout << "Token (" << b << "," << s << ") -> ";
            for (std::size_t d = 0; d < dim; ++d) {
                std::cout << static_cast<float>(y->data(b, s, d))
                          << (d + 1 == dim ? '\n' : ' ');
            }
        }
    }

    // Backward pass is removed.

    std::cout << "Embedding demo finished.\n";
    return 0;
}