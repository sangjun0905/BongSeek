#ifndef GQA_ATTENTION_HPP
#define GQA_ATTENTION_HPP

#include "Module.hpp"      // Module, Linear, Parameter 정의 포함
#include "Core.hpp"        // Variable, Function, Add, Mul 등 포함
#include "Linear.hpp"
#include "Softmax.hpp"
#include "RoPE.hpp"
#include <cmath>           // std::sqrt 사용
#include <vector>

namespace bs {

class GQAAttention : public Module {
private:
    int num_heads_;
    int num_kv_heads_;
    int head_dim_;
    int kv_repeats_;
    string name;

    std::shared_ptr<Linear> WQ_;
    std::shared_ptr<Linear> WK_;
    std::shared_ptr<Linear> WV_;
    std::shared_ptr<Linear> WO_;
    
    std::shared_ptr<RoPE> rope_;

public:
    GQAAttention() {};

    GQAAttention(const string& prefix, int input_dim, int num_heads, int num_kv_heads, int head_dim)
        : name(prefix), num_heads_(num_heads), num_kv_heads_(num_kv_heads), head_dim_(head_dim) {
        
        if (num_heads % num_kv_heads != 0) {
            throw std::invalid_argument("num_heads must be divisible by num_kv_heads");
        }
        kv_repeats_ = num_heads / num_kv_heads;

        int q_dim = num_heads * head_dim;
        int kv_dim = num_kv_heads * head_dim;

        WQ_ = std::make_shared<Linear>(input_dim, q_dim, false); // bias=false
        WK_ = std::make_shared<Linear>(input_dim, kv_dim, false);
        WV_ = std::make_shared<Linear>(input_dim, kv_dim, false);
        WO_ = std::make_shared<Linear>(q_dim, input_dim, false);

        add_module("WQ", WQ_);
        add_module("WK", WK_);
        add_module("WV", WV_);
        add_module("WO", WO_);
        
        rope_ = std::make_shared<RoPE>(head_dim_);
    }

    // Note: The signature of forward is changed to accept start_pos for RoPE
    // This means it no longer strictly overrides Module::forward.
    // For it to be a proper module, we might need a wrapper or a different design.
    // For now, let's provide the original override and a new method.

    std::shared_ptr<Variable> forward(const std::shared_ptr<Variable>& x) override {
        return forward(x, 0);
    }

    std::shared_ptr<Variable> forward(const std::shared_ptr<Variable>& x, int start_pos) {
        auto B = x->shape()[0];
        auto S = x->shape()[1];

        auto xq = (*WQ_)(x);
        auto xk = (*WK_)(x);
        auto xv = (*WV_)(x);

        xq->data = nb::reshape(xq->data, {B, S, num_heads_, head_dim_});
        xk->data = nb::reshape(xk->data, {B, S, num_kv_heads_, head_dim_});
        xv->data = nb::reshape(xv->data, {B, S, num_kv_heads_, head_dim_});
        
        xq = rope_->forward(xq, start_pos);
        xk = rope_->forward(xk, start_pos);

        auto xk_repeated = Variable::create(nb::repeat(xk->data, kv_repeats_, 2));
        auto xv_repeated = Variable::create(nb::repeat(xv->data, kv_repeats_, 2));

        auto q = nb::transposed(xq->data, {0, 2, 1, 3}); // (B, nh, S, hd)
        auto k = nb::transposed(xk_repeated->data, {0, 2, 1, 3}); // (B, nh, S, hd)
        auto v = nb::transposed(xv_repeated->data, {0, 2, 1, 3}); // (B, nh, S, hd)

        auto scores = matmul(Variable::create(q), Variable::create(nb::transposed(k, {0, 1, 3, 2})));
        scores = mul(scores, Variable::create(nb::array({1.0f / std::sqrt(head_dim_)})));
        
        scores = softmax(scores, 3);

        auto output = matmul(scores, Variable::create(v)); // (B, nh, S, hd)
        auto output_transposed = nb::transposed(output->data, {0, 2, 1, 3});
        auto contiguous_output = nb::contiguous(output_transposed);
        auto reshaped_output = nb::reshape(contiguous_output, {B, S, -1});

        return (*WO_)(Variable::create(reshaped_output));

        WQ_= 
    }

};

} // namespace bs

#endif // GQA_ATTENTION_HPP
