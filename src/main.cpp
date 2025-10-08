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
    TensorData idx_data(idx_shape);
    idx_data.fill(0.0f);

    idx_data(0, 0, 0) = 1;
    idx_data(0, 1, 0) = 3;
    idx_data(0, 2, 0) = 2;
    idx_data(1, 0, 0) = 4;
    idx_data(1, 1, 0) = 0;
    idx_data(1, 2, 0) = 1;

    auto x = Variable::create(idx_data, "token_indices");

    auto y = (*model)(x);

    std::cout << "Embedding output shape: "
              << y->data.shape_string() << "\n";
    for (std::size_t b = 0; b < batch; ++b) {
        for (std::size_t s = 0; s < seq; ++s) {
            std::cout << "Token (" << b << "," << s << ") -> ";
            for (std::size_t d = 0; d < dim; ++d) {
                std::cout << y->data(b, s, d)
                          << (d + 1 == dim ? '\n' : ' ');
            }
        }
    }

    auto grad_out = nb::Tensor<TensorValueType, TensorRank>(y->data.getShape());
    grad_out.fill(1.0f);
    y->grad = Variable::create(grad_out, "grad_out");

    y->backward();

    if (auto w_grad = model->weight()->grad) {
        std::cout << "Weight grad shape: "
                  << w_grad->data.shape_string() << "\n";
    }

    std::cout << "Embedding demo finished.\n";
    return 0;
}
