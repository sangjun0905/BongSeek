#pragma once

#include "module.hpp" // Module 클래스 정의
#include "core.hpp"   // TensorData, Variable, Parameter, Function 정의
#include <memory>
#include <stdexcept>
#include <array>

namespace bs { 

// 1. Conv1d Function (연산 노드): 실제 합성곱 연산을 담당

class Conv1d : public Function {
private:
    // 생성자에서 설정된 연산 파라미터 (stride, padding, groups)
    int stride_, padding_, groups_;

public:
    explicit Conv1d(int stride = 1, int padding = 0, int groups = 1)
        : stride_(stride), padding_(padding), groups_(groups) {}

    // 순전파: nb::conv1d를 사용하여 실제 텐서 연산을 수행합니다.
    /*std::vector<Tensor> forward(const std::vector<Tensor>& xs) override {
        // xs[0] = x (입력), xs[1] = w (가중치)
        return { nb::conv1d(xs[0], xs[1], stride_, padding_, groups_) };
    }*/
};

// Function Wrapper: conv1d(x, w, ...) 형태로 편리하게 연산을 호출하고 그래프를 연결합니다.
inline std::shared_ptr<Variable> conv1d(
    const std::shared_ptr<Variable>& x, 
    const std::shared_ptr<Variable>& w, 
    int stride = 1, 
    int padding = 0, 
    int groups = 1) 
{
    auto f = std::make_shared<Conv1d>(stride, padding, groups);
    // (*f) 오버로딩을 통해 Function::operator() 호출
    auto outs = (*f)(std::vector<std::shared_ptr<Variable>>{x, w}); 
    return std::make_shared<Variable>(outs[0]);
}


// 2. Conv1dLayer Module (계층): 가중치와 편향을 관리
class ConvldLayer : public Module {
private:
    std::shared_ptr<Parameter> W; // 학습 가능한 가중치 커널
    std::shared_ptr<Parameter> b; // 학습 가능한 편향
    int stride_, padding_;
    bool use_bias_;

public:
    // 생성자: 가중치와 편향을 초기화하고 등록합니다.
    ConvldLayer(int in_channels, int out_channels, int kernel_size, int stride = 1, int padding = 0, bool bias = true)
        : stride_(stride), padding_(padding), use_bias_(bias) 
    {
        /*
        // 1. 가중치 W 초기화 및 등록 (W Shape: {Out_C, In_C, Kernel_Size})
        // C++17 이후 std::array를 TensorShape 대신 사용했다고 가정
        std::array<size_t, 3> w_shape = {(size_t)out_channels, (size_t)in_channels, (size_t)kernel_size};
        
        // nb::randn: 랜덤 값으로 채워진 TensorData 생성 (추론 시 이 공간에 가중치가 로드됨)
        Tensor W_data = nb::randn(w_shape); 
        W = Parameter::create(W_data, "weight");
        register_parameter("weight", W);

        // 2. 편향 b 초기화 및 등록 (b Shape: {Out_C, 1, 1} - 브로드캐스팅용)
        if (use_bias_) {
            std::array<size_t, 3> b_shape = {(size_t)out_channels, 1, 1};
            Tensor b_data = nb::randn(b_shape);
            b = Parameter::create(b_data, "bias");
            register_parameter("bias", b);
        }*/
    }

    // 순전파: 실제 연산을 conv1d 래퍼 함수에 위임합니다.
    /*
    std::shared_ptr<Variable> forward(const std::shared_ptr<Variable>& x) override {
        // 1. 합성곱 연산: convld(입력 x, 가중치 W)
        auto output = conv1d(x, W, stride_, padding_); // groups는 1로 가정

        // 2. 편향 덧셈 (브로드캐스팅 연산자 오버로딩 사용)
        if (use_bias_) {
            output = output + b; // Variable + Parameter 연산자 오버로딩
        }

        return output;
    }*/

    void loadWeights(std::istream& file, const MetadataMap& metadata)
    {
        MetadataMap w1_meta;
        MetadataMap w2_meta;
        MetadataMap w3_meta;

        for(auto& [key, value] : metadata) {
            if(key.compare(0,3, "w1.") == 0) {
                w1_meta[key.substr(3)] = value; // "w1." 제외
            } 
            else if (key.compare(0,3, "w2.") == 0) {
                w2_meta[key.substr(3)] = value; // "w2." 제외
            } 
            else if (key.compare(0,3, "w3.") == 0) {
                w3_meta[key.substr(3)] = value; // "w3." 제외
            }
        }

        // conv layer 수정 하면 loadweights 호출
    }
    
};

} // namespace bs