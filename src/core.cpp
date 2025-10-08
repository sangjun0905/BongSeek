// dezero_classes.hpp
#pragma once

#include <memory>
#include <vector>
#include <algorithm>
#include <set>
#include <string>
#include <functional>
#include <iostream>
#include <Numbong.hpp> // 가정: nb::Array, nb::ones_like, nb::sum_to, nb::reshape, nb::transpose, nb::pow, nb::as_array
#include "Core.hpp"

struct Config {
    static inline bool enable_backprop = true;
    static inline bool train = true;
};

class UsingConfig {
public:
    UsingConfig(bool& flag, bool val) : flagRef(flag), old(flag) { flagRef = val; }
    ~UsingConfig() { flagRef = old; }
private:
    bool& flagRef;
    bool old;
};

inline UsingConfig no_grad() { return UsingConfig(Config::enable_backprop, false); }
inline UsingConfig test_mode() { return UsingConfig(Config::train, false); }

// Forward declare
class Function;
class Variable;

// -----------------------------
// Variable class
// -----------------------------
class Variable : public std::enable_shared_from_this<Variable> {
public:
    nb::Array data;
    std::string name;
    std::shared_ptr<Variable> grad;                 // gradient Variable
    std::shared_ptr<Function> creator;              // who created this
    int generation = 0;

    Variable() = default;
    explicit Variable(const nb::Array& arr, const std::string& n = "") : data(arr), name(n) {}

    // convenience factory
    static std::shared_ptr<Variable> create(const nb::Array& arr, const std::string& n="") {
        return std::make_shared<Variable>(arr, n);
    }//메모리 안정성을 위해서 일반 포인터대신 shared_ptr반환 new로 생성안하고 create함수로 생성

    // basic properties nb의 함수를 편의함수로 사용 getter라고 생각
    auto shape() const { return data.shape(); }
    int ndim() const { return data.ndim(); }
    size_t size() const { return data.size(); }
    auto dtype() const { return data.dtype(); }

    void set_creator(const std::shared_ptr<Function>& f);
    void unchain() { creator.reset(); }
    void cleargrad() { grad.reset(); }

    // backward (역전파)
    void backward(bool retain_grad=false, bool create_graph=false);

    // break graph from inputs upward
    void unchain_backward();

    // tensor ops that call nb-backed helpers
    std::shared_ptr<Variable> reshape(const std::vector<int>& shape) const;
    std::shared_ptr<Variable> transpose(const std::vector<int>* axes = nullptr) const;
    std::shared_ptr<Variable> T() const { return transpose(nullptr); }
    std::shared_ptr<Variable> sum(int axis = -1, bool keepdims=false) const;

    void to_cpu() { data = nb::as_cpu(data); }
    void to_gpu() { data = nb::as_gpu(data); }

    void print(const std::string& prefix="") const {
        std::cout << prefix << "Variable(name=" << name << ", shape=" << data.shape_string() << ")\n";
    }
};

// -----------------------------
// Function class (base, abstract)
// -----------------------------
class Function : public std::enable_shared_from_this<Function> {
public:
    std::vector<std::shared_ptr<Variable>> inputs;
    std::vector<std::weak_ptr<Variable>> outputs; // outputs을 약한 포인터로 설정해서 output의 참조수 추가 안됨
    int generation = 0;

    virtual ~Function() = default;

    // call: given Variable shared_ptr(s), run forward and wrap outputs
    std::vector<std::shared_ptr<Variable>> operator()(const std::vector<std::shared_ptr<Variable>>& in_vars) {
     
        inputs = in_vars;//varaible로 감싸진 객체리스트를 전달받음


        std::vector<nb::Array> xs;//거기서 nb::array만 뽑아낼건데 
        xs.reserve(inputs.size());//미리 데이터를 확보해서 재할당 연산 줄임
        for (auto &v : inputs) xs.push_back(v->data);//inputs의 포인터를 v선언해서 잠시 저장하고 v->data를 xs에 push

        // forward
        std::vector<nb::Array> ys = forward(xs);

        // wrap outputs
        std::vector<std::shared_ptr<Variable>> out_vars;
        out_vars.reserve(ys.size());//마찬가지로 미리 outputs데이터 확보
        for (auto &y : ys) {
            out_vars.push_back(Variable::create(nb::as_array(y)));
        }

        if (Config::enable_backprop) {
            // generation is max generation among inputs
            generation = 0;
            for (auto &v : inputs) generation = std::max(generation, v->generation);

            // set creator and store weak references
            for (auto &out : out_vars) {
                out->set_creator(shared_from_this());
                outputs.push_back(out);
            }//outputs 자체를 weak reference로 선언해서 참조값이 증가하지 않음
        }

        return out_vars;
    }//Function에서 주로 forward를 수행하고 이때 out->set_creator, f.outputs, f.inputs 설정해서 연결 켜버림

    //가상함수(virtual)을 설정해서 이 클래스를 상속받는 클래스 mul, sin, add등 에서 반드시 구현해야된다는 뜻 (=0) 
    //런타임에 어떤함수를 호출할지를 결정
    virtual std::vector<nb::Array> forward(const std::vector<nb::Array>& xs) = 0;

    virtual std::vector<std::shared_ptr<Variable>> backward(const std::vector<std::shared_ptr<Variable>>& gys) = 0;
};

// -----------------------------
// Implement Variable methods
// -----------------------------
inline void Variable::set_creator(const std::shared_ptr<Function>& f) {
    creator = f;
    generation = f->generation + 1;
}

inline void Variable::backward(bool retain_grad, bool create_graph) {
    if (!grad) {            //retrain_grad: 기울리 정보를 메모리에 보존할지 여부
        grad = Variable::create(nb::ones_like(data));
    }//grad가 없다면 1로 채움 data와 같은 형태로

    std::vector<std::shared_ptr<Function>> funcs;
    std::set<Function*> seen;//한번 방문한 함수 저장해서 중복 방지

    //auto변수 선언시 초기화값보고 타입을 컴파일러가 자동추론
    //[&]: 참조캡처, [=]: 값캡처
    //add_func는 backward안에 정의 된 함수여서 익명함수로 정의
    //형식 -> [] (매개변수) { // 함수 동작 } (호출 시 인자) ; 
    //std::vector.begin().end()
    auto add_func = [&](const std::shared_ptr<Function>& f) {
        if (!f) return;
        if (seen.insert(f.get()).second) {
            funcs.push_back(f);
            std::sort(funcs.begin(), funcs.end(), [](const std::shared_ptr<Function>& a, const std::shared_ptr<Function>& b){
                return a->generation < b->generation;
            });
        }
    };

    add_func(creator);

    while (!funcs.empty()) {
        auto f = funcs.back(); funcs.pop_back();
        //std::vector<std::shared_ptr<Function>> funcs

        // gather gys (one per output)
        std::vector<std::shared_ptr<Variable>> gys;
        for (auto &w : f->outputs) {//output을 약한포인터로 지정
            //std::weak_ptr.lock 은 std::weak_ptr가 가리키던 객체가 메모리에서 아직 살아있는지 확인
            //살아있다면 std::shared_ptr을 반환, 사라졌다면 nullptr을 반환
            if (auto outp = w.lock()) gys.push_back(outp->grad);
            else gys.push_back(nullptr);
        }

        // temporarily change enable_backprop according to create_graph
        {
            UsingConfig tmp(Config::enable_backprop, create_graph);
            auto gxs = f->backward(gys); // returns vector<Variable_ptr> for inputs

            // accumulate grads into inputs
            for (size_t i = 0; i < f->inputs.size(); ++i) {
                auto x = f->inputs[i];
                auto gx = (i < gxs.size() ? gxs[i] : nullptr);//std::shared_ptr<Variable>타입의 gx
                /*
                if(i < gxs.size()) gx = gxs[i];
                else gx = nullptr;
                */
                if (!gx) continue;
                
                if (!x->grad) x->grad = gx; //x->grad가 없으면 gx넣고
                else x->grad->data = x->grad->data + gx->data;//있으면 기존 grad->data에 추가

                if (x->creator) add_func(x->creator);//x->creator가 있으면 add_func
            }
        }

        if (!retain_grad) {
            for (auto &w : f->outputs) {
                if (auto outp = w.lock()) outp->grad.reset();
            }//역전파가 완료된 이후 불필요해진 기울기 reset
        }
    }
}

inline void Variable::unchain_backward() {
    if (!creator) return;
    std::vector<std::shared_ptr<Function>> funcs;
    funcs.push_back(creator);
    while (!funcs.empty()) {
        auto f = funcs.back(); funcs.pop_back();
        for (auto &x : f->inputs) {
            if (x->creator) {
                funcs.push_back(x->creator);
                x->unchain();
            }
        }
    }
}

class Parameter : public Variable{
    public:
        Parameter() = default;
        explicit Parameter(const nb::Array& arr, const std::string& n ="")
            : Variable(arr,n){

            }

            static std::shardd_ptr<Parameter> create(const nb::Array& arr, const std::string& n=""){
                return std::make_shared<Parameter>(arr,n);
            }
};

// -----------------------------
// Some concrete Function implementations
// -----------------------------
class Add : public Function {
    nb::Shape x0_shape, x1_shape;
public:
    std::vector<nb::Array> forward(const std::vector<nb::Array>& xs) override {
        x0_shape = xs[0].shape();
        x1_shape = xs[1].shape();
        return { xs[0] + xs[1] };
    }

    std::vector<std::shared_ptr<Variable>> backward(const std::vector<std::shared_ptr<Variable>>& gys) override {
        auto gy = gys[0];
        auto gx0 = Variable::create(gy->data);
        auto gx1 = Variable::create(gy->data);
        if (!(x0_shape == x1_shape)) {
            gx0->data = nb::sum_to(gx0->data, x0_shape);
            gx1->data = nb::sum_to(gx1->data, x1_shape);
        }
        return { gx0, gx1 };
    }
};

class Mul : public Function {
public:
    std::vector<nb::Array> forward(const std::vector<nb::Array>& xs) override {
        return { xs[0] * xs[1] };
    }

    std::vector<std::shared_ptr<Variable>> backward(const std::vector<std::shared_ptr<Variable>>& gys) override {
        auto x0 = inputs[0];
        auto x1 = inputs[1];
        auto gy = gys[0];
        auto gx0 = Variable::create(gy->data * x1->data);
        auto gx1 = Variable::create(gy->data * x0->data);
        if (!(x0->data.shape() == x1->data.shape())) {
            gx0->data = nb::sum_to(gx0->data, x0->data.shape());
            gx1->data = nb::sum_to(gx1->data, x1->data.shape());
        }
        return { gx0, gx1 };
    }
};

class Neg : public Function {
public:
    std::vector<nb::Array> forward(const std::vector<nb::Array>& xs) override {
        return { -xs[0] };
    }
    std::vector<std::shared_ptr<Variable>> backward(const std::vector<std::shared_ptr<Variable>>& gys) override {
        auto gy = gys[0];
        return { Variable::create(-gy->data) };
    }
};

class Sub : public Function {
    nb::Shape x0_shape, x1_shape;
public:
    std::vector<nb::Array> forward(const std::vector<nb::Array>& xs) override {
        x0_shape = xs[0].shape();
        x1_shape = xs[1].shape();
        return { xs[0] - xs[1] };
    }
    std::vector<std::shared_ptr<Variable>> backward(const std::vector<std::shared_ptr<Variable>>& gys) override {
        auto gy = gys[0];
        auto gx0 = Variable::create(gy->data);
        auto gx1 = Variable::create(-gy->data);
        if (!(x0_shape == x1_shape)) {
            gx0->data = nb::sum_to(gx0->data, x0_shape);
            gx1->data = nb::sum_to(gx1->data, x1_shape);
        }
        return { gx0, gx1 };
    }
};

class Div : public Function {
public:
    std::vector<nb::Array> forward(const std::vector<nb::Array>& xs) override {
        return { xs[0] / xs[1] };
    }
    std::vector<std::shared_ptr<Variable>> backward(const std::vector<std::shared_ptr<Variable>>& gys) override {
        auto x0 = inputs[0];
        auto x1 = inputs[1];
        auto gy = gys[0];
        auto gx0 = Variable::create(gy->data / x1->data);
        auto gx1 = Variable::create(gy->data * ( - x0->data / (x1->data * x1->data) ));
        if (!(x0->data.shape() == x1->data.shape())) {
            gx0->data = nb::sum_to(gx0->data, x0->data.shape());
            gx1->data = nb::sum_to(gx1->data, x1->data.shape());
        }
        return { gx0, gx1 };
    }
};

class Pow : public Function {
    double c;
public:
    explicit Pow(double cc) : c(cc) {}
    std::vector<nb::Array> forward(const std::vector<nb::Array>& xs) override {
        return { nb::pow(xs[0], c) };
    }
    std::vector<std::shared_ptr<Variable>> backward(const std::vector<std::shared_ptr<Variable>>& gys) override {
        auto x = inputs[0];
        auto gy = gys[0];
        auto gx = Variable::create(c * nb::pow(x->data, c - 1.0) * gy->data);
        return { gx };
    }
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
