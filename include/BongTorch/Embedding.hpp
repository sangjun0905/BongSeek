#ifndef EMBEDDING_HPP
#define EMBEDDING_HPP
#pragma once

#include "Module.hpp"
#include "Core.hpp"
#include <memory>
#include <stdexcept>
#include <vector>

namespace bs {

class EmbeddingFunction : public Function {
private:
    std::size_t batch_size_{0};
    std::size_t seq_len_{0};
    std::size_t embedding_dim_{0};
    std::size_t vocab_size_{0};
    std::vector<std::size_t> flat_indices_;

public:
    std::vector<TensorData> forward(const std::vector<TensorData>& xs) override {
        const auto& indices = xs[0];
        const auto& weight = xs[1];

        auto idx_shape = indices.getShape();
        if (idx_shape[2] != 1) {
            throw std::runtime_error("Embedding indices tensor must have shape (B, S, 1).");
        }

        auto weight_shape = weight.getShape();
        batch_size_ = idx_shape[0];
        seq_len_ = idx_shape[1];
        vocab_size_ = weight_shape[0];
        embedding_dim_ = weight_shape[1];

        TensorShape out_shape = {batch_size_, seq_len_, embedding_dim_};
        TensorData output(out_shape);

        flat_indices_.resize(batch_size_ * seq_len_);

        for (std::size_t b = 0; b < batch_size_; ++b) {
            for (std::size_t s = 0; s < seq_len_; ++s) {
                float raw = indices(b, s, 0);
                if (raw < 0.0f) {
                    throw std::runtime_error("Embedding index must be non-negative.");
                }
                std::size_t idx = static_cast<std::size_t>(raw);
                if (idx >= vocab_size_) {
                    throw std::runtime_error("Embedding index out of range.");
                }

                flat_indices_[b * seq_len_ + s] = idx;

                for (std::size_t d = 0; d < embedding_dim_; ++d) {
                    output(b, s, d) = weight(idx, d, 0);
                }
            }
        }

        return {output};
    }

    std::vector<std::shared_ptr<Variable>> backward(const std::vector<std::shared_ptr<Variable>>& gys) override {
        auto gy = gys[0];

        TensorShape idx_shape = inputs[0]->data.getShape();
        TensorData grad_indices(idx_shape);
        grad_indices.fill(0.0f); // 정수 인덱스이므로 기울기는 모두 0

        TensorShape weight_shape = inputs[1]->data.getShape();
        TensorData grad_weight(weight_shape);
        grad_weight.fill(0.0f);

        for (std::size_t b = 0; b < batch_size_; ++b) {
            for (std::size_t s = 0; s < seq_len_; ++s) {
                std::size_t idx = flat_indices_[b * seq_len_ + s];
                for (std::size_t d = 0; d < embedding_dim_; ++d) {
                    grad_weight(idx, d, 0) += gy->data(b, s, d);
                }
            }
        }

        return {
            Variable::create(grad_indices, "embedding_indices_grad"),
            Variable::create(grad_weight, "embedding_weight_grad")
        };
    }
};

inline std::shared_ptr<Variable> embedding_lookup(const std::shared_ptr<Variable>& indices,
                                                  const std::shared_ptr<Variable>& weight) {
    auto f = std::make_shared<EmbeddingFunction>();
    auto outs = (*f)(std::vector<std::shared_ptr<Variable>>{indices, weight});
    return outs[0];
}

class Embedding : public Module {
private:
    std::shared_ptr<Parameter> weight_;
    std::size_t vocab_size_{0};
    std::size_t embedding_dim_{0};

public:
    Embedding(std::size_t vocab_size, std::size_t embedding_dim)
        : vocab_size_(vocab_size), embedding_dim_(embedding_dim) {
        TensorShape weight_shape = {vocab_size_, embedding_dim_, 1};
        TensorData weight_data(weight_shape);
        weight_data.fill(0.0f); // 필요하면 랜덤 초기화로 교체

        weight_ = Parameter::create(weight_data, "weight");
        register_parameter("weight", weight_);
    }

    std::shared_ptr<Variable> forward(const std::shared_ptr<Variable>& x) override {
        auto f = std::make_shared<EmbeddingFunction>();
        auto outs = (*f)(std::vector<std::shared_ptr<Variable>>{x, weight_});
        return outs[0];
    }

    std::shared_ptr<Parameter> weight() const { return weight_; }
    std::size_t vocab_size() const { return vocab_size_; }
    std::size_t embedding_dim() const { return embedding_dim_; }
};

} // namespace bs
#endif