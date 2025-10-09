#pragma once

#include "Core.hpp"
#include <memory>
#include <stdexcept>
#include <vector>

namespace bs {

class Embedding : public Module {
private:
    std::shared_ptr<Parameter> W;
    std::size_t vocab_size_;
    std::size_t dim_;

public:
    Embedding(std::size_t vocab_size, std::size_t dim)
        : vocab_size_(vocab_size), dim_(dim) {
        
        TensorShape weight_shape = {vocab_size, dim, 1};
        auto weight_tensor = Tensor(weight_shape);
        W = Parameter::create(weight_tensor, "W");
        register_parameter("W", W);
    }

    std::shared_ptr<Variable> forward(const std::shared_ptr<Variable>& x) override {
        const auto& indices = x->data;
        const auto& weight = W->data;

        auto idx_shape = indices.shape();
        if (idx_shape.size() != 3 || idx_shape[2] != 1) {
            throw std::runtime_error("Indices tensor must have shape (B, S, 1).");
        }
        
        auto weight_shape = weight.shape();
        auto batch_size_ = idx_shape[0];
        auto seq_len_ = idx_shape[1];
        auto vocab_size_ = weight_shape[0];
        auto embedding_dim_ = weight_shape[1];

        TensorShape out_shape = {batch_size_, seq_len_, embedding_dim_};
        Tensor output(out_shape);

        for (std::size_t b = 0; b < batch_size_; ++b) {
            for (std::size_t s = 0; s < seq_len_; ++s) {
                float raw = static_cast<float>(indices(b, s, 0)); 
                if (raw < 0.0f) {
                    throw std::runtime_error("Embedding index must be non-negative.");
                }
                std::size_t idx = static_cast<std::size_t>(raw);
                if (idx >= vocab_size_) {
                    throw std::runtime_error("Embedding index out of range.");
                }

                for (std::size_t d = 0; d < embedding_dim_; ++d) {
                    output(b, s, d) = weight(idx, d, 0);
                }
            }
        }

        return Variable::create(output, "embedding_output");
    }

    std::shared_ptr<Parameter> weight() const { return W; }
};

} // namespace bs