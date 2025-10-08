#ifndef MODULE_HPP
#define MODULE_HPP

#include "core.hpp" // Variable, Parameter, Function 정의를 가져옵니다.
#include <map>
#include <vector>
#include <memory>

// NN 모듈의 기본 클래스 역할을 합니다 (PyTorch의 nn.Module과 유사).
class Module : public std::enable_shared_from_this<Module> {
private:
    // 모듈이 소유한 파라미터(가중치)를 이름으로 관리합니다.
    std::map<std::string, std::shared_ptr<Parameter>> parameters;

    // 모듈이 소유한 하위 모듈을 이름으로 관리합니다.
    std::map<std::string, std::shared_ptr<Module>> children;

public:
    virtual ~Module() = default;

    // 모든 모듈은 이 순전파 인터페이스를 구현해야 합니다.
    virtual std::shared_ptr<Variable> forward(const std::shared_ptr<Variable>& x) = 0;

    // 사용자 호출을 위한 Operator Overloading
    std::shared_ptr<Variable> operator()(const std::shared_ptr<Variable>& x) {
        return forward(x);
    }

    // --- 관리 기능 ---

    // Parameter 등록 (가중치 관리)
    void register_parameter(const std::string& name, const std::shared_ptr<Parameter>& param) {
        parameters[name] = param;
    }
    
    // 하위 Module 등록 (모델 계층 구조를 위해)
    void add_module(const std::string& name, const std::shared_ptr<Module>& module) {
        children[name] = module;
    }

    // 이 모듈과 하위 모듈의 모든 파라미터를 재귀적으로 수집합니다.
    std::vector<std::shared_ptr<Parameter>> get_parameters() {
        std::vector<std::shared_ptr<Parameter>> all_params;
        
        // 1. 자신의 파라미터 추가
        for (const auto& pair : parameters) {
            all_params.push_back(pair.second);
        }

        // 2. 하위 모듈의 파라미터 재귀적으로 추가
        for (const auto& pair : children) {
            auto child_params = pair.second->get_parameters();
            all_params.insert(all_params.end(), child_params.begin(), child_params.end());
        }
        
        return all_params;
    }
};

// ----------------------------------------------------
// MatMul Function 정의 (Linear 계층에 필요)
// ----------------------------------------------------

class MatMul : public Function {
public:
    // nb::Array 대신 TensorData 타입을 사용합니다. (core.hpp의 using에 따름)
    std::vector<TensorData> forward(const std::vector<TensorData>& xs) override {
        // nb::Tensor::matmul을 사용하여 행렬 곱셈을 수행합니다.
        return { xs[0].matmul(xs[1]) };
    }
    // 추론 전용이므로 backward는 정의하지 않습니다.
};

inline std::shared_ptr<Variable> matmul(const std::shared_ptr<Variable>& a, const std::shared_ptr<Variable>& b) {
    auto f = std::make_shared<MatMul>();
    auto outs = (*f)({a, b});
    return outs[0]; // Function::operator()의 반환 타입에 맞춤
}

// ----------------------------------------------------
// 구체적인 Linear 계층 구현 (Module 상속)
// ----------------------------------------------------

class Linear : public Module {
private:
    std::shared_ptr<Parameter> W; // 가중치 행렬
    std::shared_ptr<Parameter> b; // 편향 벡터 (옵션)
    bool use_bias;

public:
    Linear(size_t in_size, size_t out_size, bool bias = true) : use_bias(bias) {
        // 1. 가중치 초기화 (임의의 더미 텐서 데이터로 가정합니다)
        // Rank 3 (Batch, Out, In) 또는 (Out, In, 1) 등의 형식을 가정합니다.
        TensorData W_data(out_size, in_size, 1); 
        
        W = Parameter::create(W_data, "weight");
        register_parameter("weight", W);

        if (use_bias) {
            TensorData b_data(out_size, 1, 1); 
            b = Parameter::create(b_data, "bias");
            register_parameter("bias", b);
        }
    }

    std::shared_ptr<Variable> forward(const std::shared_ptr<Variable>& x) override {
        // Linear 연산: y = x @ W^T + b
        
        // 1. W^T: 가중치 W를 transpose 합니다.
        auto W_t = Variable::create(W->data.transpose(), "W_t"); 
        
        // 2. 행렬 곱: x @ W^T
        auto output = matmul(x, W_t);
        
        // 3. 편향 덧셈 (브로드캐스팅 필요)
        if (use_bias) {
            // Variable의 + 연산자 오버로딩 (Add Function) 사용
            output = output + b; 
        }
        
        return output;
    }
};

#endif // MODULE_HPP