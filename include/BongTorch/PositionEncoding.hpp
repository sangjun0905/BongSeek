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

    Tensor create_pe_array(int S) {
        Tensor P({1, (size_t)S, (size_t)d_model});

        for (int pos = 0; pos < S; ++pos) {
            for (int i = 0; i < d_model / 2; ++i) {
                double div_term = std::pow(10000.0, (double)(2 * i) / d_model);
                double sin_val = std::sin((double)pos / div_term);
                double cos_val = std::cos((double)pos / div_term);
                
                P(0, pos, 2 * i) = nb::BFloat16(sin_val);
                if (2 * i + 1 < d_model) {
                    P(0, pos, 2 * i + 1) = nb::BFloat16(cos_val);
                }
            }
        }
        return P;
    }

    std::vector<Tensor> forward(const std::vector<Tensor>& xs) override {
        Tensor X = xs[0];
        int S = X.shape()[1]; // Get sequence length from input
        Tensor P = create_pe_array(S);

        // Ensure P can be broadcast to X's shape if batch size > 1
        // (Current implementation assumes P is broadcastable)
        Tensor Y = X + P;

        return { Y };
    }
};

inline std::shared_ptr<Variable> position_encoding(const std::shared_ptr<Variable>& x, int max_len, int d_model) {
    auto f = std::make_shared<PositionEncoding>(max_len, d_model);
    auto outs = (*f)(std::vector<std::shared_ptr<Variable>>{x});
    return outs;
}

} // namespace bs
