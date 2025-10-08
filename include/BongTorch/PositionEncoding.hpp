#pragma once

#include "Core.hpp"
#include <cmath>

namespace bs {

class PositionEncoding : public Function {
private:
    int max_len; // 최대 시퀀스 길이
    int d_model; // 모델 차원 (d_model)

public:
    PositionEncoding(int max_len, int d_model) : max_len(max_len), d_model(d_model) {}

    TensorData create_pe_array(const TensorData& input_data) {
        int S = input_data.shape()[1]; // 입력 텐서의 시퀀스 길이

        TensorData P(Shape({(size_t)S, (size_t)d_model}));

        for (int pos = 0; pos < S; ++pos) {
            for (int i = 0; i < d_model; ++i) {
                double freq = 1.0 / std::pow(10000.0, (double)i / d_model);
                // This is a dummy implementation for setting values.
                // if (i % 2 == 0) {
                //     P.set_value({(size_t)pos, (size_t)i}, std::sin(pos * freq));
                // } else {
                //     P.set_value({(size_t)pos, (size_t)i}, std::cos(pos * freq));
                // }
            }
        }

        return P;
    }

    std::vector<TensorData> forward(const std::vector<TensorData>& xs) override {
        TensorData X = xs[0];
        TensorData P = create_pe_array(X);

        TensorData Y = X + P;

        return { Y };
    }

    std::vector<std::shared_ptr<Variable>> backward(const std::vector<std::shared_ptr<Variable>>& gys) override {
        auto gy = gys[0];
        return { Variable::create(gy->data) };
    }
};

inline std::shared_ptr<Variable> position_encoding(const std::shared_ptr<Variable>& x, int max_len, int d_model) {
    auto f = std::make_shared<PositionEncoding>(max_len, d_model);
    auto outs = (*f)(std::vector<std::shared_ptr<Variable>>{x});
    return outs[0];
}

} // namespace bs
