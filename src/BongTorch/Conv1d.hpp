#ifndef CONV_LAYER_HPP
#define CONV_LAYER_HPP

#include "module.hpp" // Module, Parameter 정의 및 matmul, Linear 등 포함
#include "core.hpp"   // Variable, Function, Add, Mul 등 정의 포함

class Conv1D : public Function {
private:
    using TensorData = typename Variable::TensorData; 
    int stride_, padding_, groups_; 

public:
    // 일반적인 Conv1D를 위해 groups의 기본값을 1로 설정합니다.
    explicit Conv1D(int stride = 1, int padding = 0, int groups = 1) 
        : stride_(stride), padding_(padding), groups_(groups) {}

    std::vector<TensorData> forward(const std::vector<TensorData>& xs) override {
        // xs[0]: 입력 (Input, X)
        // xs[1]: 가중치 (Weight, W)
        
        // nb::conv1d(input, weight, stride, padding, groups)를 NumBong에 정의한다고 가정합니다.
        return { nb::conv1d(xs[0], xs[1], stride_, padding_, groups_) }; 
    }
};

inline std::shared_ptr<Variable> conv1d(
    const std::shared_ptr<Variable>& x, 
    const std::shared_ptr<Variable>& w, 
    int stride = 1, 
    int padding = 0, // 기본 패딩은 0으로 설정
    int groups = 1) 
{
    auto f = std::make_shared<Conv1D>(stride, padding, groups);
    auto outs = (*f)({x, w});
    return outs[0];
}



class Conv1DLayer : public Module {
private:
    std::shared_ptr<Parameter> W; // 가중치 텐서 (Filter)
    std::shared_ptr<Parameter> b; // 편향 텐서 (Bias)
    int stride_, padding_;
    bool use_bias_;

public:
    // 생성자: (입력 채널 수, 출력 채널 수, 커널 크기)를 받습니다.
    Conv1DLayer(int in_channels, int out_channels, int kernel_size, int stride = 1, int padding = 0, bool bias = true)
        : stride_(stride), padding_(padding), use_bias_(bias) 
    {
        // 1. 가중치 초기화 및 등록
        // 필터 셰이프: [C_out, C_in, K]
        TensorData W_data(out_channels, in_channels, kernel_size); 
        W = Parameter::create(W_data, "weight");
        register_parameter("weight", W);

        // 2. 편향 초기화 및 등록
        if (use_bias_) {
            // 편향 셰이프: [C_out, 1, 1] (브로드캐스팅을 위해)
            TensorData b_data(out_channels, 1, 1);
            b = Parameter::create(b_data, "bias");
            register_parameter("bias", b);
        }
    }

    // Module::forward 구현
    std::shared_ptr<Variable> forward(const std::shared_ptr<Variable>& x) override {
        // x의 shape은 (Batch, Input_Channels, Seq_Length)를 가정합니다. 
        
        // 1. 컨볼루션 연산: conv1d(x, W, groups=1)
        // groups가 기본값 1이므로 일반 컨볼루션이 실행됩니다.
        auto output = conv1d(x, W, stride_, padding_);

        // 2. 편향 덧셈 (브로드캐스팅)
        if (use_bias_) {
            // Variable의 + 연산자 오버로딩을 통해 편향 덧셈 (브로드캐스팅)
            output = output + b; 
        }

        return output;
    }
};