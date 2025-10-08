#ifndef CORE_HPP
#define CORE_HPP

#include <memory>
#include <vector>
#include <algorithm>
#include <set>
#include <string>
#include <functional>
#include <iostream>
//#include <Numbong.hpp>

struct Config {
    static inline bool enable_backprop = true;
    static inline bool train = true;
};

class UsingConfig {
public:
    UsingConfig(bool& flag, bool val) ;
    ~UsingConfig() ;
private:
    bool& flagRef;
    bool old;
};

inline UsingConfig no_grad(); 
inline UsingConfig test_mode();

class Function;
class Variable{
    public:
        nb::Array data;
        std::string name;
        std::shared_ptr<Variable> grad;
        std::shared_ptr<Function> creator;
        int generation = 0;
 
        Variable() = default;
        explicit Variable(const nb::Array& arr, const std::string& n = "");

        static std::shared_ptr<Variable> create(const nb::Array& arr, const std::string& n="");
        auto shape() const;
        int ndim() const;
        size_t size() const;
        auto dtype() const;

        void set_creator(const std::shared_ptr<Function>& f);
        void unchain() ;
        void cleargrad() ;

    // backward (역전파)
        void backward(bool retain_grad=false, bool create_graph=false);

    // break graph from inputs upward
        void unchain_backward();

    // tensor ops that call nb-backed helpers
        std::shared_ptr<Variable> reshape(const std::vector<int>& shape) const;
        std::shared_ptr<Variable> transpose(const std::vector<int>* axes = nullptr) const;
        std::shared_ptr<Variable> T() const;
        std::shared_ptr<Variable> sum(int axis = -1, bool keepdims=false) const;

        void to_cpu();
        void to_gpu();

        void print(const std::string& prefix="");
}
        

class Function : public std::enable_shared_from_this<Function> {
public:
    std::vector<std::shared_ptr<Variable>> inputs;
    std::vector<std::weak_ptr<Variable>> outputs; // outputs을 약한 포인터로 설정해서 output의 참조수 추가 안됨
    int generation = 0;

    virtual ~Function() = default;

    // __call__
    std::vector<std::shared_ptr<Variable>> operator()(const std::vector<std::shared_ptr<Variable>>& in_vars) 
    //Function에서 주로 forward를 수행하고 이때 out->set_creator, f.outputs, f.inputs 설정해서 연결 켜버림

    //가상함수(virtual)을 설정해서 이 클래스를 상속받는 클래스 mul, sin, add등 에서 반드시 구현해야된다는 뜻 (=0) 
    //런타임에 어떤함수를 호출할지를 결정
    virtual std::vector<nb::Array> forward(const std::vector<nb::Array>& xs) = 0;

    virtual std::vector<std::shared_ptr<Variable>> backward(const std::vector<std::shared_ptr<Variable>>& gys) = 0;
};

class Parameter : public Variable{
    public:
        Parameter() = default;
        explicit Parameter(const nb::Array& arr, const std::string& n ="");
            
        static std::shardd_ptr<Parameter> create(const nb::Array& arr, const std::string& n="")
};

class Add : public Function {
    nb::Shape x0_shape, x1_shape;
public:
    std::vector<nb::Array> forward(const std::vector<nb::Array>& xs);
    std::vector<std::shared_ptr<Variable>> backward(const std::vector<std::shared_ptr<Variable>>& gys);
     };

class Mul : public Function {
public:
    std::vector<nb::Array> forward(const std::vector<nb::Array>& xs);
    std::vector<std::shared_ptr<Variable>> backward(const std::vector<std::shared_ptr<Variable>>& gys);
    };

class Neg : public Function {
public:
    std::vector<nb::Array> forward(const std::vector<nb::Array>& xs);
    std::vector<std::shared_ptr<Variable>> backward(const std::vector<std::shared_ptr<Variable>>& gys);
};

class Sub : public Function {
    nb::Shape x0_shape, x1_shape;
public:
    std::vector<nb::Array> forward(const std::vector<nb::Array>& xs);   
    std::vector<std::shared_ptr<Variable>> backward(const std::vector<std::shared_ptr<Variable>>& gys); 
};

class Div : public Function {
public:
    std::vector<nb::Array> forward(const std::vector<nb::Array>& xs); 
    std::vector<std::shared_ptr<Variable>> backward(const std::vector<std::shared_ptr<Variable>>& gys);
        
};

class Pow : public Function {
    double c;
public:
    explicit Pow(double cc);
    std::vector<nb::Array> forward(const std::vector<nb::Array>& xs);
    std::vector<std::shared_ptr<Variable>> backward(const std::vector<std::shared_ptr<Variable>>& gys);
};    

// -----------------------------
// Operator helpers: friend-like free functions
// -----------------------------
inline std::shared_ptr<Variable> add(const std::shared_ptr<Variable>& a, const std::shared_ptr<Variable>& b) {
    auto f = std::make_shared<Add>();//add객체를 f에 포인터로 저장
    auto outs = (*f)(std::vector<std::shared_ptr<Variable>>{a, b});//a,b데이터만 추출해서 덧셈,,연산결과를 담는 새로운 Variable객체 생성, 계산 그래프 기록
    return outs[0];//덧셈은 결과가 하나이므로
}
inline std::shared_ptr<Variable> operator+(const std::shared_ptr<Variable>& a, const std::shared_ptr<Variable>& b){
    return add(a,b);
} //이제 덧셈연산을 a+b로 할 수 있게됨


inline std::shared_ptr<Variable> mul(const std::shared_ptr<Variable>& a, const std::shared_ptr<Variable>& b) {
    auto f = std::make_shared<Mul>();
    auto outs = (*f)(std::vector<std::shared_ptr<Variable>>{a, b});
    return outs[0];
}
inline std::shared_ptr<Variable> operator*(const std::shared_ptr<Variable>& a, const std::shared_ptr<Variable>& b){
    return mul(a,b);
} //이제 곱셈연산을 a*b로 할 수 있게됨


inline std::shared_ptr<Variable> neg(const std::shared_ptr<Variable>& a) {
    auto f = std::make_shared<Neg>();
    auto outs = (*f)(std::vector<std::shared_ptr<Variable>>{a});
    return outs[0];
}
inline std::shared_ptr<Variable> operator-(const std::shared_ptr<Variable>& a, const std::shared_ptr<Variable>& b){
    return neg(a);
} //이제 부정연산을 -a 로 할 수 있게됨

inline std::shared_ptr<Variable> sub(const std::shared_ptr<Variable>& a, const std::shared_ptr<Variable>& b) {
    auto f = std::make_shared<Sub>();
    auto outs = (*f)(std::vector<std::shared_ptr<Variable>>{a, b});
    return outs[0];
}
inline std::shared_ptr<Variable> operator-(const std::shared_ptr<Variable>& a, const std::shared_ptr<Variable>& b){
    return sub(a,b);
}//이제 덧셈연산을 a-b로 할 수 있게됨

inline std::shared_ptr<Variable> divv(const std::shared_ptr<Variable>& a, const std::shared_ptr<Variable>& b) {
    auto f = std::make_shared<Div>();
    auto outs = (*f)(std::vector<std::shared_ptr<Variable>>{a, b});
    return outs[0];
}
inline std::shared_ptr<Variable> operator/(const std::shared_ptr<Variable>& a, const std::shared_ptr<Variable>& b){
    return divv(a,b);
}//이제 나눗셈연산을 a/b로 할 수 있게됨

inline std::shared_ptr<Variable> powv(const std::shared_ptr<Variable>& a, double c) {
    auto f = std::make_shared<Pow>(c);
    auto outs = (*f)(std::vector<std::shared_ptr<Variable>>{a});
    return outs[0];
}
inline std::shared_ptr<Variable> operator^(const std::shared_ptr<Variable>& a, const std::shared_ptr<Variable>& b){
    return powv(a,b);
}//이제 거듭제곱연산을 a^b로 할 수 있게됨


// -----------------------------
// Simple demo
// -----------------------------
inline void demo() {
    nb::Array a = nb::array({1.0, 2.0, 3.0});
    nb::Array b = nb::array({4.0, 5.0, 6.0});
    auto A = Variable::create(a, "A");
    auto B = Variable::create(b, "B");

    // C = A * B + A
    auto C = add(mul(A, B), A);
    C->print("C: ");
    C->backward();
    if (A->grad) {
        std::cout << "A.grad shape: " << A->grad->data.shape_string() << "\n";
    } else {
        std::cout << "A.grad none\n";
    }
}

// namespace dz

#endif 