#pragma once

#include "Core.hpp"
#include <cmath>
#include <memory>

namespace bs {

class Softmax : public Function {
private:
    int axis_;

public:
    explicit Softmax(int axis = -1) : axis_(axis) {}

    std::vector<Tensor> forward(const std::vector<Tensor>& xs) override {
        const Tensor& x = xs[0];

        int axis = axis_;
        if (axis < 0) {
            axis += static_cast<int>(x.ndim());
        }
        if (axis != 2) {
            throw std::invalid_argument("Softmax: only axis=2 is supported.");
        }

        const auto shape = x.getShape();
        const std::size_t batch = shape[0];
        const std::size_t seq = shape[1];
        const std::size_t dim = shape[2];

        Tensor y(shape);

        for (std::size_t b = 0; b < batch; ++b) {
            for (std::size_t s = 0; s < seq; ++s) {
                float max_val = static_cast<float>(x(b, s, 0));
                for (std::size_t d = 1; d < dim; ++d) {
                    max_val = std::max(max_val,
                                        static_cast<float>(x(b, s, d)));
                }

                float sum = 0.0f;
                for (std::size_t d = 0; d < dim; ++d) {
                    const float shifted =
                        static_cast<float>(x(b, s, d)) - max_val;
                    const float ex = std::exp(shifted);
                    y(b, s, d) = static_cast<TensorValueType>(ex);
                    sum += ex;
                }

                const float inv_sum = (sum > 0.0f) ? 1.0f / sum : 0.0f;
                for (std::size_t d = 0; d < dim; ++d) {
                    const float v = static_cast<float>(y(b, s, d));
                    y(b, s, d) = static_cast<TensorValueType>(v * inv_sum);
                }
            }
        }

        return { y };
    }
};

inline std::shared_ptr<Variable> softmax(const std::shared_ptr<Variable>& x,
                                         int axis = -1) {
    auto f = std::make_shared<Softmax>(axis);
    // (*f) 오버로딩을 통해 Function 호출
    auto outs = (*f)(std::vector<std::shared_ptr<Variable>>{x});
    return outs; 
}

} // namespace bs
