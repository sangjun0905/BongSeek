#ifndef CONV_LAYER_HPP
#define CONV_LAYER_HPP

#include "module.hpp"
#include "core.hpp"

namespace bs { // Add bs namespace

class Conv1D : public Function {
private:
    // using TensorData = typename Variable::TensorData; // This is now in core.hpp
    int stride_, padding_, groups_;

public:
    explicit Conv1D(int stride = 1, int padding = 0, int groups = 1)
        : stride_(stride), padding_(padding), groups_(groups) {}

    std::vector<TensorData> forward(const std::vector<TensorData>& xs) override {
        return { nb::conv1d(xs[0], xs[1], stride_, padding_, groups_) };
    }

    std::vector<std::shared_ptr<Variable>> backward(const std::vector<std::shared_ptr<Variable>>& gys) override {
        throw std::runtime_error("Backward for Conv1D is not implemented.");
        return {};
    }
};

inline std::shared_ptr<Variable> conv1d(
    const std::shared_ptr<Variable>& x, 
    const std::shared_ptr<Variable>& w, 
    int stride = 1, 
    int padding = 0, 
    int groups = 1) 
{
    auto f = std::make_shared<Conv1D>(stride, padding, groups);
    auto outs = (*f)({x, w});
    return outs[0];
}

class Conv1DLayer : public Module {
private:
    std::shared_ptr<Parameter> W; 
    std::shared_ptr<Parameter> b; 
    int stride_, padding_;
    bool use_bias_;

public:
    Conv1DLayer(int in_channels, int out_channels, int kernel_size, int stride = 1, int padding = 0, bool bias = true)
        : stride_(stride), padding_(padding), use_bias_(bias) 
    {
        std::array<size_t, 3> w_shape = {(size_t)out_channels, (size_t)in_channels, (size_t)kernel_size};
        TensorData W_data = nb::randn(w_shape);
        W = Parameter::create(W_data, "weight");
        register_parameter("weight", W);

        if (use_bias_) {
            std::array<size_t, 3> b_shape = {(size_t)out_channels, 1, 1};
            TensorData b_data = nb::randn(b_shape);
            b = Parameter::create(b_data, "bias");
            register_parameter("bias", b);
        }
    }

    std::shared_ptr<Variable> forward(const std::shared_ptr<Variable>& x) override {
        auto output = conv1d(x, W, stride_, padding_);

        if (use_bias_) {
            output = output + b;
        }

        return output;
    }
};

} // namespace bs

#endif // CONV_LAYER_HPP
