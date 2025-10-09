#ifndef CORE_HPP
#define CORE_HPP

#include <algorithm>
#include <array>
#include <functional>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include "../NumBong/Tensor.hpp" // NumBong Tensor 클래스 정의

// --- 타입 별칭 및 설정 ---
using TensorValueType = float;
constexpr std::size_t TensorRank = 3; 

// Core에서 사용할 텐서 타입을 nb::Tensor<float, 3>으로 명확히 정의
using Tensor = nb::Tensor<TensorValueType, TensorRank>; 
using TensorShape = typename Tensor::shape_type;
using Shape = nb::Shape; // nb::Shape은 std::vector<size_t>

// --- NumBong 연산 래퍼 함수 ---
// Tensor의 멤버 함수(add, mul 등)를 호출하여 전역 함수처럼 사용
namespace nb {
inline Tensor add(const Tensor& a, const Tensor& b) { return a.add(b); }
inline Tensor mul(const Tensor& a, const Tensor& b) { return a.mul(b); }
inline Tensor neg(const Tensor& a) { return a.negate(); }
inline Tensor sub(const Tensor& a, const Tensor& b) { return a.sub(b); }
inline Tensor div(const Tensor& a, const Tensor& b) { return a.div(b); }
inline Tensor pow(const Tensor& a, double exponent) { return a.pow(exponent); }
inline const Tensor& as_array(const Tensor& a) { return a; } // Identity 함수로 가정
}

// --- Function 및 Variable 정의 ---

class Function; // Variable 클래스보다 먼저 선언됨

class Variable : public std::enable_shared_from_this<Variable> {
public:
    Tensor data;
    std::string name;
    // 역전파 관련 멤버 (grad, creator, generation)는 추론 전용이므로 제거됨

    explicit Variable(const Tensor& arr, const std::string& n = "") : data(arr), name(n) {}

    static std::shared_ptr<Variable> create(const Tensor& arr, const std::string& n = "") {
        return std::make_shared<Variable>(arr, n);
    }
    
    // Tensor 멤버 함수에 맞게 수정됨
    Shape shape() const { return data.shape_vector(); } 
    int ndim() const { return static_cast<int>(data.ndim()); } 
    std::size_t size() const { return data.size(); } 
    auto dtype() const { return data.dtype(); }
};

class Function : public std::enable_shared_from_this<Function> {
public:
    // 역전파 관련 멤버 (inputs, outputs 등)는 추론 전용이므로 제거됨
    
    virtual ~Function() = default;

    // operator() 오버로딩: Variable 입력을 받아 Tensor::forward를 호출하고 Variable 출력을 반환
    std::shared_ptr<Variable> operator()(const std::vector<std::shared_ptr<Variable>>& in_vars) {
        std::vector<Tensor> xs;
        xs.reserve(in_vars.size());
        for (const auto& v : in_vars) {
            xs.push_back(v->data); // Variable의 텐서만 뽑아서 담음
        }

        std::vector<Tensor> ys = forward(xs);
        
        // Function은 오직 하나의 Variable을 출력한다고 가정
        auto out = Variable::create(ys[0]); 
        
        // 역전파 연결 로직 제거됨 (추론 전용)
        return out; 
    }

    // forward는 Tensor 기반으로 오버라이딩합니다.
    virtual std::vector<Tensor> forward(const std::vector<Tensor>& xs) = 0;
    
    // backward는 추론 전용이므로 제거됨 (각 Function 구현체에서 재정의 필요)
};

class Parameter : public Variable {
public:
    explicit Parameter(const Tensor& arr, const std::string& n = "") : Variable(arr, n) {}

    static std::shared_ptr<Parameter> create(const Tensor& arr, const std::string& n = "") {
        return std::make_shared<Parameter>(arr, n);
    }
};

// --- 추론 전용 Function 구현 ---
class Add : public Function {
public:
    std::vector<Tensor> forward(const std::vector<Tensor>& xs) override {
        return { nb::add(xs[0], xs[1]) };
    }
};

class Mul : public Function {
public:
    std::vector<Tensor> forward(const std::vector<Tensor>& xs) override {
        return { nb::mul(xs[0] ,xs[1]) };
    }
};

class Neg : public Function {
public:
    std::vector<Tensor> forward(const std::vector<Tensor>& xs) override {
        return { nb::neg(xs[0]) };
    }
};

class Sub : public Function {
public:
    std::vector<Tensor> forward(const std::vector<Tensor>& xs) override {
        return { nb::sub(xs[0] ,xs[1]) };
    }
};

class Div : public Function {
public:
    std::vector<Tensor> forward(const std::vector<Tensor>& xs) override {
        return { nb::div(xs[0] ,xs[1]) };
    }
};

class Pow : public Function {
    double c;
public:
    explicit Pow(double cc) : c(cc) {}

    std::vector<Tensor> forward(const std::vector<Tensor>& xs) override {
        return { nb::pow(xs[0], c) };
    }
};

// --- 연산자 래퍼 함수 (Wrapper Functions) ---

inline std::shared_ptr<Variable> apply_op(const std::shared_ptr<Variable>& a, const std::shared_ptr<Variable>& b, const std::shared_ptr<Function>& f) {
    auto out = (*f)(std::vector<std::shared_ptr<Variable>>{a, b});
    return out;
}

inline std::shared_ptr<Variable> apply_op(const std::shared_ptr<Variable>& a, const std::shared_ptr<Function>& f) {
    auto out = (*f)(std::vector<std::shared_ptr<Variable>>{a});
    return out;
}

inline std::shared_ptr<Variable> add(const std::shared_ptr<Variable>& a, const std::shared_ptr<Variable>& b) {
    return apply_op(a, b, std::make_shared<Add>());
}


inline std::shared_ptr<Variable> mul(const std::shared_ptr<Variable>& a, const std::shared_ptr<Variable>& b) {
    return apply_op(a, b, std::make_shared<Mul>());
}


inline std::shared_ptr<Variable> neg(const std::shared_ptr<Variable>& a) {
    return apply_op(a, std::make_shared<Neg>());
}


inline std::shared_ptr<Variable> sub(const std::shared_ptr<Variable>& a, const std::shared_ptr<Variable>& b) {
    return apply_op(a, b, std::make_shared<Sub>());
}


inline std::shared_ptr<Variable> divv(const std::shared_ptr<Variable>& a, const std::shared_ptr<Variable>& b) {
    return apply_op(a, b, std::make_shared<Div>());
}

inline std::shared_ptr<Variable> powv(const std::shared_ptr<Variable>& a, double c) {
    auto f = std::make_shared<Pow>(c);
    auto outs = (*f)(std::vector<std::shared_ptr<Variable>>{a});
    return outs;
}

#endif