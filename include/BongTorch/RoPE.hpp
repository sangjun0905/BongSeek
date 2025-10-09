#pragma once

#include "Core.hpp"
#include <memory>

namespace bs {

class RoPE : public Function {
private:
    static Tensor apply_rope(const Tensor& x,
                             const Tensor& cos_tensor,
                             const Tensor& sin_tensor) {
        const int last_axis = static_cast<int>(x.ndim()) - 1;
        const int d_k = static_cast<int>(x.shape()[last_axis]);
        const int d_half = d_k / 2;

        Tensor x_a = nb::split(x, 0, d_half);
        Tensor x_b = nb::split(x, 1, d_half);

        Tensor term1 = (x_a * cos_tensor) - (x_b * sin_tensor);
        Tensor term2 = (x_b * cos_tensor) + (x_a * sin_tensor);

        std::vector<Tensor> parts{term1, term2};
        return nb::concat(parts, last_axis);
    }

public:
    std::shared_ptr<Variable> forward(const std::shared_ptr<Variable>& x,
                                      const std::shared_ptr<Variable>& cos_tensor,
                                      const std::shared_ptr<Variable>& sin_tensor) {
        Tensor rotated = apply_rope(x->data, cos_tensor->data, sin_tensor->data);
        return Variable::create(rotated, "rope");
    }

    std::vector<Tensor> forward(const std::vector<Tensor>& xs) override {
        return { apply_rope(xs[0], xs[1], xs[2]) };
    }
};

inline std::shared_ptr<Variable> rope(const std::shared_ptr<Variable>& x,
                                      const std::shared_ptr<Variable>& cos_tensor,
                                      const std::shared_ptr<Variable>& sin_tensor) {
    auto f = std::make_shared<RoPE>();
    return f->forward(x, cos_tensor, sin_tensor);
}

} // namespace bs
