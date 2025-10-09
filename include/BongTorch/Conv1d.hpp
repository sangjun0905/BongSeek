#pragma once

#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

#include "Core.hpp"

namespace bs {
 
namespace detail {

inline Tensor conv1d_forward(const Tensor& x,
                                 const Tensor& w,
                                 int stride,
                                 int padding,
                                 std::size_t groups) {
    const auto x_shape = x.getShape();
    const auto w_shape = w.getShape();

    const std::size_t B   = x_shape[0];
    const std::size_t Cin = x_shape[1];
    const std::size_t Sin = x_shape[2];
    const std::size_t Cout = w_shape[0];
    const std::size_t K    = w_shape[2];

    if (groups == 0 || Cin % groups != 0 || Cout % groups != 0) {
        throw std::runtime_error("conv1d_forward: invalid groups for given channel sizes.");
    }
    const std::size_t Cin_per_g  = Cin / groups;
    const std::size_t Cout_per_g = Cout / groups;
    if (w_shape[1] != Cin_per_g) {
        throw std::runtime_error("conv1d_forward: weight shape[1] must equal Cin_per_group.");
    }

    const std::size_t Sout = static_cast<std::size_t>(
        (static_cast<int>(Sin) + 2 * padding - static_cast<int>(K)) / stride + 1);

    Tensor out({B, Cout, Sout});

    for (std::size_t b = 0; b < B; ++b) {
        for (std::size_t g = 0; g < groups; ++g) {
            const std::size_t ic0 = g * Cin_per_g;
            const std::size_t oc0 = g * Cout_per_g;

            for (std::size_t ocg = 0; ocg < Cout_per_g; ++ocg) {
                const std::size_t oc = oc0 + ocg;

                for (std::size_t t = 0; t < Sout; ++t) {
                    nb::BFloat16 acc(0.0f);

                    for (std::size_t icg = 0; icg < Cin_per_g; ++icg) {
                        const std::size_t ic = ic0 + icg;

                        for (std::size_t k = 0; k < K; ++k) {
                            const int x_idx = static_cast<int>(t * stride + k) - padding;
                            if (x_idx < 0 || x_idx >= static_cast<int>(Sin)) {
                                continue;
                            }

                            acc += x(b, ic, static_cast<std::size_t>(x_idx)) * w(oc, icg, k);
                        }
                    }

                    out(b, oc, t) = acc;
                }
            }
        }
    }

    return out;
}

inline Tensor linear_forward(const Tensor& input,
                                 const Tensor& weight) {
    const auto in_shape = input.getShape();
    const auto w_shape  = weight.getShape();

    const std::size_t B   = in_shape[0];
    const std::size_t S   = in_shape[1];
    const std::size_t Cin = in_shape[2];
    const std::size_t Cout = w_shape[0];

    if (w_shape[1] != Cin) {
        throw std::runtime_error("linear_forward: weight in_features must match input.");
    }

    Tensor out({B, S, Cout});

    for (std::size_t b = 0; b < B; ++b) {
        for (std::size_t s = 0; s < S; ++s) {
            for (std::size_t of = 0; of < Cout; ++of) {
                nb::BFloat16 acc(0.0f);
                for (std::size_t inf = 0; inf < Cin; ++inf) {
                    acc += input(b, s, inf) * weight(of, inf, 0);
                }
                out(b, s, of) = acc;
            }
        }
    }

    return out;
}

inline Tensor transpose_bc(const Tensor& tensor) {
    const auto shape = tensor.getShape();
    Tensor out({shape[0], shape[2], shape[1]});
    for (std::size_t b = 0; b < shape[0]; ++b)
        for (std::size_t c = 0; c < shape[1]; ++c)
            for (std::size_t s = 0; s < shape[2]; ++s)
                out(b, s, c) = tensor(b, c, s);
    return out;
}

inline Tensor transpose_cs(const Tensor& tensor) {
    const auto shape = tensor.getShape();
    Tensor out({shape[0], shape[2], shape[1]});
    for (std::size_t b = 0; b < shape[0]; ++b)
        for (std::size_t s = 0; s < shape[1]; ++s)
            for (std::size_t c = 0; c < shape[2]; ++c)
                out(b, c, s) = tensor(b, s, c);
    return out;
}

} // namespace detail

class Conv1dFunction : public Function {
public:
    explicit Conv1dFunction(int stride, int padding, std::size_t groups)
        : stride_(stride), padding_(padding), groups_(groups) {}

    std::vector<Tensor> forward(const std::vector<Tensor>& xs) override {
        if (xs.size() != 2) {
            throw std::runtime_error("Conv1dFunction expects [input, weight].");
        }
        return { detail::conv1d_forward(xs[0], xs[1], stride_, padding_, groups_) };
    }

private:
    int stride_;
    int padding_;
    std::size_t groups_;
};

inline std::shared_ptr<Variable> conv1d_op(const std::shared_ptr<Variable>& x,
                                           const std::shared_ptr<Variable>& weight,
                                           int stride,
                                           int padding,
                                           std::size_t groups) {
    auto f = std::make_shared<Conv1dFunction>(stride, padding, groups);
    auto outs = (*f)(std::vector<std::shared_ptr<Variable>>{x, weight});
    return outs;
}

class Conv1d : public Module {
public:
    Conv1d(){};

    Conv1d(std::size_t in_channels,
           std::size_t conv_out_channels,
           std::size_t kernel,
           std::size_t in_proj_out_features,
           std::size_t out_proj_out_features,
           int stride = 1,
           int padding = 0,
           std::size_t groups = 1)
        : stride_(stride),
          padding_(padding),
          groups_(groups),
          in_channels_(in_channels),
          conv_out_channels_(conv_out_channels),
          in_proj_out_features_(in_proj_out_features),
          out_proj_out_features_(out_proj_out_features) {

        if (groups_ == 0 || in_channels_ % groups_ != 0) {
            throw std::runtime_error("Conv1d: invalid groups for given input channels.");
        }
        if (conv_out_channels_ % groups_ != 0) {
            throw std::runtime_error("Conv1d: conv_out_channels must be divisible by groups.");
        }
        if (out_proj_out_features_ == 0) {
            throw std::runtime_error("Conv1d: out_proj_out_features must be positive.");
        }

        std::size_t gating_factor = 0;
        if (in_proj_out_features_ % out_proj_out_features_ == 0) {
            gating_factor = in_proj_out_features_ / out_proj_out_features_;
        }
        if (gating_factor == 0) {
            throw std::runtime_error("Conv1d: in_proj_out_features must be a multiple of out_proj_out_features.");
        }
        if (gating_factor != 1 && gating_factor != 3) {
            throw std::runtime_error("Conv1d: only gating factors of 1 or 3 are supported.");
        }

        use_gated_path_ = (gating_factor == 3);
        out_proj_input_features_ = use_gated_path_ ? out_proj_out_features_ : in_proj_out_features_;

        TensorShape conv_shape = {conv_out_channels_, in_channels_ / groups_, kernel};
        conv_weight_ = Parameter::create(Tensor(conv_shape), "conv.weight");
        conv_weight_->data.fill(nb::BFloat16(0.0f));
        register_parameter("conv.weight", conv_weight_);

        TensorShape in_proj_shape = {in_proj_out_features_, conv_out_channels_, 1};
        in_proj_weight_ = Parameter::create(Tensor(in_proj_shape), "in_proj.weight");
        in_proj_weight_->data.fill(nb::BFloat16(0.0f));
        register_parameter("in_proj.weight", in_proj_weight_);

        TensorShape out_proj_shape = {out_proj_out_features_, out_proj_input_features_, 1};
        out_proj_weight_ = Parameter::create(Tensor(out_proj_shape), "out_proj.weight");
        out_proj_weight_->data.fill(nb::BFloat16(0.0f));
        register_parameter("out_proj.weight", out_proj_weight_);
    }

    std::shared_ptr<Variable> forward(const std::shared_ptr<Variable>& x) override {
        auto conv_input = adapt_input_layout(x);

        auto conv_out = conv1d_op(conv_input, conv_weight_, stride_, padding_, groups_);

        auto conv_transposed = detail::transpose_bc(conv_out->data);
        auto in_proj_out = detail::linear_forward(conv_transposed, in_proj_weight_->data);

        Tensor proj_input = use_gated_path_ ? apply_gating(in_proj_out)
                                                : in_proj_out;

        auto out_proj_out = detail::linear_forward(proj_input, out_proj_weight_->data);

        auto final_back = detail::transpose_cs(out_proj_out);
        return Variable::create(final_back, "conv_out_proj");
    }

    std::shared_ptr<Parameter> conv_weight() const { return conv_weight_; }
    std::shared_ptr<Parameter> in_proj_weight() const { return in_proj_weight_; }
    std::shared_ptr<Parameter> out_proj_weight() const { return out_proj_weight_; }
    void loadWeights(std::istream& file, const MetadataMap& metadata)
    {
        MetadataMap conv_meta;
        MetadataMap in_proj_meta;
        MetadataMap out_proj_meta;

        for(auto& [key, value] : metadata) {
            if(key.compare(0,5, "conv.") == 0) {
                conv_meta[key.substr(5)] = value; // "w1." 제외
            } 
            else if (key.compare(0, 8, "in_proj.") == 0) {
                in_proj_meta[key.substr(8)] = value; // "w2." 제외
            } 
            else if (key.compare(0,9, "out_proj.") == 0) {
                out_proj_meta[key.substr(9)] = value; // "w3." 제외
            }
        }

        // conv layer 수정 하면 loadweights 호출
    }

private:
    std::shared_ptr<Variable> adapt_input_layout(const std::shared_ptr<Variable>& x) const {
        const auto shape = x->data.getShape();
        if (shape[1] == in_channels_) {
            return x;
        }

        if (shape[2] == in_channels_) {
            auto transposed = detail::transpose_cs(x->data);
            return Variable::create(transposed, x->name + "_transpose_cs");
        }

        throw std::runtime_error("Conv1d::forward: input tensor must be (B, C_in, S) or (B, S, C_in).");
    }

    Tensor apply_gating(const Tensor& in_proj_out) const {
        const auto shape = in_proj_out.getShape();
        const std::size_t B = shape[0];
        const std::size_t S = shape[1];
        const std::size_t block = out_proj_out_features_;

        Tensor gated({B, S, block});
        const nb::BFloat16 one(1.0f);

        for (std::size_t b = 0; b < B; ++b) {
            for (std::size_t s = 0; s < S; ++s) {
                for (std::size_t i = 0; i < block; ++i) {
                    const nb::BFloat16 a = in_proj_out(b, s, i);
                    const nb::BFloat16 b_gate = in_proj_out(b, s, i + block);
                    const nb::BFloat16 c = in_proj_out(b, s, i + 2 * block);
                    const nb::BFloat16 exp_neg = nb::bfloat16_exp(-b_gate);
                    const nb::BFloat16 denom = one + exp_neg;
                    const nb::BFloat16 sigma = one / denom;
                    gated(b, s, i) = a * sigma + c;
                }
            }
        }

        return gated;
    }

    std::shared_ptr<Parameter> conv_weight_;
    std::shared_ptr<Parameter> in_proj_weight_;
    std::shared_ptr<Parameter> out_proj_weight_;

    int stride_;
    int padding_;
    std::size_t groups_;

    std::size_t in_channels_;
    std::size_t conv_out_channels_;
    std::size_t in_proj_out_features_;
    std::size_t out_proj_out_features_;
    std::size_t out_proj_input_features_;
    bool use_gated_path_ = false;

    
};

} // namespace bs
