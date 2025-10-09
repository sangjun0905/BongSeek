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

#include "../NumBong/Tensor.hpp" 

using TensorValueType = float;
constexpr std::size_t TensorRank = 3;
using TensorData = nb::Tensor<TensorValueType, TensorRank>;  // -> TensorData라고 별칭 지정
using Shape = nb::Shape;                                     // ->Shape이라고 별칭 지정(자료형)
using TensorShape = typename TensorData::shape_type;         // ->Tensor내부에서 공식적으로 사용하는  shape타입 별칭

// 역전파 관련 Config, UsingConfig 등 모두 제거됨.

class Function;

class Variable : public std::enable_shared_from_this<Variable> {
public:
    TensorData data;
    std::string name;
    // 역전파 관련 멤버 제거: grad, creator, generation

    explicit Variable(const TensorData& arr, const std::string& n = "") : data(arr), name(n) {}

    static std::shared_ptr<Variable> create(const TensorData& arr, const std::string& n = "") {
        return std::make_shared<Variable>(arr, n);
    }//-> 이시점에 tensorA는 이미 만들어진 TensorData

    Shape shape() const { return data.shape_vector(); }         //-> shape반환 ex) {2,3,4}
    int ndim() const { return static_cast<int>(data.ndim()); }  //-> rank반환
    size_t size() const { return data.size(); }                 //-> 전체 요소수 반환
    auto dtype() const { return data.dtype(); }                 //-> TensorValueType반환
} ;

class Function : public std::enable_shared_from_this<Function> {
public:
    // 역전파 관련 inputs, outputs, generation 제거
    
    virtual ~Function() = default;

    // operator()는 단일 Variable 출력을 반환하도록 수정
    std::shared_ptr<Variable> operator()(const std::vector<std::shared_ptr<Variable>>& in_vars) {
        std::vector<TensorData> xs;   
        xs.reserve(in_vars.size());
        for (const auto& v : in_vars) {
            xs.push_back(v->data);         //_> Variable의 텐서만 뽑아서 담음
        }

        std::vector<TensorData> ys = forward(xs);
        
        // Function은 오직 하나의 Variable을 출력한다고 가정
        auto out = Variable::create(nb::as_array(ys[0]));
        
        return out;   //-> 최종 산출물은 Variable 포인터
            }

    // forward는 TensorData 기반으로 오버라이딩합니다.
    virtual std::vector<TensorData> forward(const std::vector<TensorData>& xs) = 0;
    
};

class Parameter : public Variable {
public:
    explicit Parameter(const TensorData& arr, const std::string& n = "") : Variable(arr, n) {}

    static std::shared_ptr<Parameter> create(const TensorData& arr, const std::string& n = "") {
        return std::make_shared<Parameter>(arr, n);
    }
};

// --- 추론 전용 Function 구현 ---
class Add : public Function {
public:
    std::vector<TensorData> forward(const std::vector<TensorData>& xs) override {
        return { nb::add(xs[0], xs[1]) };
    }    
};

class Mul : public Function {
public:
    std::vector<TensorData> forward(const std::vector<TensorData>& xs) override {
        return { nb::mul(xs[0] ,xs[1]) };
    }
};

class Neg : public Function {
public:
    std::vector<TensorData> forward(const std::vector<TensorData>& xs) override {
        return { nb::neg(xs[0]) };
    }
};

class Sub : public Function {
public:
    std::vector<TensorData> forward(const std::vector<TensorData>& xs) override {
        return { nb::sub(xs[0] ,xs[1]) };
    }
};

class Div : public Function {
public:
    std::vector<TensorData> forward(const std::vector<TensorData>& xs) override {
        return { nb::div(xs[0] ,xs[1]) };
    }
};

class Pow : public Function {
    double c;
public:
    explicit Pow(double cc) : c(cc) {}

    std::vector<TensorData> forward(const std::vector<TensorData>& xs) override {
        return { nb::pow(xs[0], c) };
    }                           //---> Function을 상속받으므로 Function의 forward오버라이딩 실제 nb::pow()연산 수행
};

// --- 연산자 오버로딩 (Operator Overloading) ---

inline std::shared_ptr<Variable> apply_op(const std::shared_ptr<Variable>& a, const std::shared_ptr<Variable>& b, const std::shared_ptr<Function>& f) {
    auto out = (*f)({a, b});  //-> <add>function객체 생성 후 a,b 넘김
    return out;
}

inline std::shared_ptr<Variable> apply_op(const std::shared_ptr<Variable>& a, const std::shared_ptr<Function>& f) {
    auto out = (*f)({a});
    return out;
}

inline std::shared_ptr<Variable> add(const std::shared_ptr<Variable>& a, const std::shared_ptr<Variable>& b) {
    return apply_op(a, b, std::make_shared<Add>()); //->2. apply_op()호출
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