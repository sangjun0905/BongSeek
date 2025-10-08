#include <iostream>
#include "NumBong/Tensor.hpp"
#include "BongTorch/Core.hpp"

int main() {
    using nb::Tensor;

    Tensor<float, 3> m1(2, 2, 2);
    Tensor<float, 3> m2(2, 2, 2);

    m1(0, 0, 0) = 1.0f;
    m1(0, 0, 1) = 2.0f;
    m1(0, 1, 0) = 3.0f;
    m1(0, 1, 1) = 4.0f;

    m2(0, 0, 0) = 5.0f;
    m2(0, 0, 1) = 6.0f;
    m2(0, 1, 0) = 7.0f;
    m2(0, 1, 1) = 8.0f;

    auto A = Variable::create(m1, "A");
    auto B = Variable::create(m2, "B");

    auto C = add(mul(A, B), A);
    C->backward();

    std::cout << "C shape: " << C->data.shape_string() << '\n';
    if (A->grad) {
        std::cout << "dC/dA(0,0,0): " << A->grad->data(0, 0, 0) << '\n';
    }

    return 0;
}
