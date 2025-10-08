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

// Tensor.hpp는 그대로 유지합니다.
#include "../NumBong/Tensor.hpp" 

using TensorValueType = float;
constexpr std::size_t TensorRank = 3;
using TensorData = nb::Tensor<TensorValueType, TensorRank>;
using Shape = nb::Shape;
using TensorShape = typename TensorData::shape_type;

// 추론 전용이므로 Config, UsingConfig, no_grad(), test_mode()는 제거합니다.

class Function;

class Variable : public std::enable_shared_from_this<Variable> {
public:
    TensorData data;
    std::string name;
    // 역전파 관련 멤버 제거: grad, creator, generation

    explicit Variable(const TensorData& arr, const std::string& n = "") : data(arr), name(n) {}

    static std::shared_ptr<Variable> create(const TensorData& arr, const std::string& n = "") {
        return std::make_shared<Variable>(arr, n);
    }

    Shape shape() const { return data.shape_vector(); }
    int ndim() const { return static_cast<int>(data.ndim()); }
    size_t size() const { return data.size(); }
    auto dtype() const { return data.dtype(); }

    // 역전파 관련 함수 제거: set_creator, unchain, cleargrad, backward, unchain_backward

    void to_cpu() { data = nb::as_cpu(data); }
    void to_gpu() { data = nb::as_gpu(data); }

    void print(const std::string& prefix="") const {
        std::cout << prefix << "Variable(name=" << name << ", shape=" << data.shape_string() << ")\n";
    }
};

class Function : public std::enable_shared_from_this<Function> {
public:
    // 추론 전용이므로 inputs, outputs, generation을 제거하거나 간소화할 수 있지만, 
    // 현재 구조에서는 forward 함수만 필요합니다.
    // inputs는 forward 호출 시 인자로 받으므로 멤버 변수로 유지할 필요가 없습니다.
    // outputs (weak_ptr)는 역전파를 위한 것이었으므로 제거합니다.

    virtual ~Function() = default;

    std::shared_ptr<Variable> operator()(const std::vector<std::shared_ptr<Variable>>& in_vars) {
        std::vector<TensorData> xs;
        xs.reserve(in_vars.size());
        for (const auto& v : in_vars) {
            xs.push_back(v->data);
        }

        std::vector<TensorData> ys = forward(xs);
        
        // Function은 오직 하나의 Variable을 출력한다고 가정하고 out_vars를 생성
        auto out = Variable::create(nb::as_array(ys[0]));
        
        // 역전파 관련 코드 제거 (generation 설정, creator 설정, outputs 저장)

        return out; // 단일 출력을 반환하도록 수정
    }

    // forward는 TensorData 기반으로 오버라이딩합니다.
    virtual std::vector<TensorData> forward(const std::vector<TensorData>& xs) = 0;
    
    // 역전파 관련 함수 제거: backward
};

// 역전파 관련 Variable::set_creator, Variable::backward, Variable::unchain_backward 정의 제거

class Parameter : public Variable {
public:
    explicit Parameter(const TensorData& arr, const std::string& n = "") : Variable(arr, n) {}

    static std::shared_ptr<Parameter> create(const TensorData& arr, const std::string& n = "") {
        return std::make_shared<Parameter>(arr, n);
    }
};

// --- 추론 전용 Function 구현 ---
// Function의 forward만 유지하고 backward를 제거합니다.
// 브로드캐스팅 처리를 위한 shape 저장은 forward에서 제거합니다.

class Add : public Function {
public:
    std::vector<TensorData> forward(const std::vector<TensorData>& xs) override {
        // nb::Tensor가 브로드캐스팅을 지원한다고 가정하고 그냥 더합니다.
        return { xs[0] + xs[1] };
    }
    // backward 제거
};

class Mul : public Function {
public:
    std::vector<TensorData> forward(const std::vector<TensorData>& xs) override {
        return { xs[0] * xs[1] };
    }
    // backward 제거
};

class Neg : public Function {
public:
    std::vector<TensorData> forward(const std::vector<TensorData>& xs) override {
        return { -xs[0] };
    }
    // backward 제거
};

class Sub : public Function {
public:
    std::vector<TensorData> forward(const std::vector<TensorData>& xs) override {
        return { xs[0] - xs[1] };
    }
    // backward 제거
};

class Div : public Function {
public:
    std::vector<TensorData> forward(const std::vector<TensorData>& xs) override {
        return { xs[0] / xs[1] };
    }
    // backward 제거
};

class Pow : public Function {
    double c;
public:
    explicit Pow(double cc) : c(cc) {}

    std::vector<TensorData> forward(const std::vector<TensorData>& xs) override {
        return { nb::pow(xs[0], c) };
    }
    // backward 제거
};

// --- 연산자 오버로딩 (Operator Overloading) ---

inline std::shared_ptr<Variable> apply_op(const std::shared_ptr<Variable>& a, const std::shared_ptr<Variable>& b, const std::shared_ptr<Function>& f) {
    auto out = (*f)({a, b});
    return out;
}

inline std::shared_ptr<Variable> apply_op(const std::shared_ptr<Variable>& a, const std::shared_ptr<Function>& f) {
    auto out = (*f)({a});
    return out;
}

inline std::shared_ptr<Variable> add(const std::shared_ptr<Variable>& a, const std::shared_ptr<Variable>& b) {
    return apply_op(a, b, std::make_shared<Add>());
}

inline std::shared_ptr<Variable> operator+(const std::shared_ptr<Variable>& a, const std::shared_ptr<Variable>& b) {
    return add(a, b);
}

inline std::shared_ptr<Variable> mul(const std::shared_ptr<Variable>& a, const std::shared_ptr<Variable>& b) {
    return apply_op(a, b, std::make_shared<Mul>());
}

inline std::shared_ptr<Variable> operator*(const std::shared_ptr<Variable>& a, const std::shared_ptr<Variable>& b) {
    return mul(a, b);
}

inline std::shared_ptr<Variable> neg(const std::shared_ptr<Variable>& a) {
    return apply_op(a, std::make_shared<Neg>());
}

inline std::shared_ptr<Variable> operator-(const std::shared_ptr<Variable>& a) {
    return neg(a);
}

inline std::shared_ptr<Variable> sub(const std::shared_ptr<Variable>& a, const std::shared_ptr<Variable>& b) {
    return apply_op(a, b, std::make_shared<Sub>());
}

inline std::shared_ptr<Variable> operator-(const std::shared_ptr<Variable>& a, const std::shared_ptr<Variable>& b) {
    return sub(a, b);
}

inline std::shared_ptr<Variable> divv(const std::shared_ptr<Variable>& a, const std::shared_ptr<Variable>& b) {
    return apply_op(a, b, std::make_shared<Div>());
}

inline std::shared_ptr<Variable> operator/(const std::shared_ptr<Variable>& a, const std::shared_ptr<Variable>& b) {
    return divv(a, b);
}

inline std::shared_ptr<Variable> powv(const std::shared_ptr<Variable>& a, double c) {
    auto f = std::make_shared<Pow>(c);
    auto outs = (*f)(std::vector<std::shared_ptr<Variable>>{a});
    return outs;
}

inline std::shared_ptr<Variable> operator^(const std::shared_ptr<Variable>& a, double c) {
    return powv(a, c);
}

inline void demo() {
    // nb::Tensor 생성자를 (3, 3, 3) 대신 3차원 배열 초기화 리스트를 사용하도록 가정
    // 또는 nb::Tensor가 (d1, d2, d3) 형태의 생성자를 가지고 있다고 가정합니다.
    TensorData a(3, 3, 3);
    TensorData b(3, 3, 3);
    a.fill(2.0f); // 값 채우기
    b.fill(3.0f); // 값 채우기

    auto A = Variable::create(a, "A");
    auto B = Variable::create(b, "B");

    // 순전파 연산만 수행
    auto C = add(mul(A, B), A);
    C->print("C: "); 
    // 역전파 관련 코드 제거: C->backward();
    // A->grad 확인 코드 제거
}

#endif