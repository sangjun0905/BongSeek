// dezero_classes.hpp
#pragma once

#include <memory>
#include <vector>
#include <algorithm>
#include <set>
#include <string>
#include <functional>
#include <iostream>
#include <bongpy.hpp> // 가정: bp::Array, bp::ones_like, bp::sum_to, bp::reshape, bp::transpose, bp::pow, bp::as_array

namespace dz {

namespace bp = bongpy;

// -----------------------------
// Config RAII helpers
// -----------------------------
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
    bp::Array data;
    std::string name;
    std::shared_ptr<Variable> grad;                 // gradient Variable
    std::shared_ptr<Function> creator;              // who created this
    int generation = 0;

    Variable() = default;
    explicit Variable(const bp::Array& arr, const std::string& n = "") : data(arr), name(n) {}

    // convenience factory
    static std::shared_ptr<Variable> create(const bp::Array& arr, const std::string& n="") {
        return std::make_shared<Variable>(arr, n);
    }//메모리 안정성을 위해서 일반 포인터대신 shared_ptr반환 new로 생성안하고 create함수로 생성

    // basic properties bp의 함수를 편의함수로 사용 getter라고 생각
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

    // tensor ops that call bp-backed helpers
    std::shared_ptr<Variable> reshape(const std::vector<int>& shape) const;
    std::shared_ptr<Variable> transpose(const std::vector<int>* axes = nullptr) const;
    std::shared_ptr<Variable> T() const { return transpose(nullptr); }
    std::shared_ptr<Variable> sum(int axis = -1, bool keepdims=false) const;

    void to_cpu() { data = bp::as_cpu(data); }
    void to_gpu() { data = bp::as_gpu(data); }

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
    std::vector<std::weak_ptr<Variable>> outputs; // keep weak refs to outputs
    int generation = 0;

    virtual ~Function() = default;

    // call: given Variable shared_ptr(s), run forward and wrap outputs
    std::vector<std::shared_ptr<Variable>> operator()(const std::vector<std::shared_ptr<Variable>>& in_vars) {
        // store inputs
        inputs = in_vars;

        // extract raw arrays
        std::vector<bp::Array> xs;
        xs.reserve(inputs.size());
        for (auto &v : inputs) xs.push_back(v->data);

        // forward
        std::vector<bp::Array> ys = forward(xs);

        // wrap outputs
        std::vector<std::shared_ptr<Variable>> out_vars;
        out_vars.reserve(ys.size());
        for (auto &y : ys) {
            out_vars.push_back(Variable::create(bp::as_array(y)));
        }

        if (Config::enable_backprop) {
            // generation is max generation among inputs
            generation = 0;
            for (auto &v : inputs) generation = std::max(generation, v->generation);

            // set creator and store weak references
            for (auto &out : out_vars) {
                out->set_creator(shared_from_this());
                outputs.push_back(out);
            }
        }

        return out_vars;
    }

    // override in derived classes
    virtual std::vector<bp::Array> forward(const std::vector<bp::Array>& xs) = 0;

    // backward receives gradient Variables for outputs and returns gradient Variables for inputs
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
    if (!grad) {
        grad = Variable::create(bp::ones_like(data));
    }//grad가 없다면 1로 채움 data와 같은 형태로

    std::vector<std::shared_ptr<Function>> funcs;
    std::set<Function*> seen;//한번 방문한 함수 저장해서 중복 방지

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

        // gather gys (one per output)
        std::vector<std::shared_ptr<Variable>> gys;
        for (auto &w : f->outputs) {
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
                auto gx = (i < gxs.size() ? gxs[i] : nullptr);
                if (!gx) continue;
                if (!x->grad) x->grad = gx;
                else x->grad->data = x->grad->data + gx->data;

                if (x->creator) add_func(x->creator);
            }
        }

        if (!retain_grad) {
            for (auto &w : f->outputs) {
                if (auto outp = w.lock()) outp->grad.reset();
            }
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

// -----------------------------
// Some concrete Function implementations
// -----------------------------
class Add : public Function {
    bp::Shape x0_shape, x1_shape;
public:
    std::vector<bp::Array> forward(const std::vector<bp::Array>& xs) override {
        x0_shape = xs[0].shape();
        x1_shape = xs[1].shape();
        return { xs[0] + xs[1] };
    }

    std::vector<std::shared_ptr<Variable>> backward(const std::vector<std::shared_ptr<Variable>>& gys) override {
        auto gy = gys[0];
        auto gx0 = Variable::create(gy->data);
        auto gx1 = Variable::create(gy->data);
        if (!(x0_shape == x1_shape)) {
            gx0->data = bp::sum_to(gx0->data, x0_shape);
            gx1->data = bp::sum_to(gx1->data, x1_shape);
        }
        return { gx0, gx1 };
    }
};

class Mul : public Function {
public:
    std::vector<bp::Array> forward(const std::vector<bp::Array>& xs) override {
        return { xs[0] * xs[1] };
    }

    std::vector<std::shared_ptr<Variable>> backward(const std::vector<std::shared_ptr<Variable>>& gys) override {
        auto x0 = inputs[0];
        auto x1 = inputs[1];
        auto gy = gys[0];
        auto gx0 = Variable::create(gy->data * x1->data);
        auto gx1 = Variable::create(gy->data * x0->data);
        if (!(x0->data.shape() == x1->data.shape())) {
            gx0->data = bp::sum_to(gx0->data, x0->data.shape());
            gx1->data = bp::sum_to(gx1->data, x1->data.shape());
        }
        return { gx0, gx1 };
    }
};

class Neg : public Function {
public:
    std::vector<bp::Array> forward(const std::vector<bp::Array>& xs) override {
        return { -xs[0] };
    }
    std::vector<std::shared_ptr<Variable>> backward(const std::vector<std::shared_ptr<Variable>>& gys) override {
        auto gy = gys[0];
        return { Variable::create(-gy->data) };
    }
};

class Sub : public Function {
    bp::Shape x0_shape, x1_shape;
public:
    std::vector<bp::Array> forward(const std::vector<bp::Array>& xs) override {
        x0_shape = xs[0].shape();
        x1_shape = xs[1].shape();
        return { xs[0] - xs[1] };
    }
    std::vector<std::shared_ptr<Variable>> backward(const std::vector<std::shared_ptr<Variable>>& gys) override {
        auto gy = gys[0];
        auto gx0 = Variable::create(gy->data);
        auto gx1 = Variable::create(-gy->data);
        if (!(x0_shape == x1_shape)) {
            gx0->data = bp::sum_to(gx0->data, x0_shape);
            gx1->data = bp::sum_to(gx1->data, x1_shape);
        }
        return { gx0, gx1 };
    }
};

class Div : public Function {
public:
    std::vector<bp::Array> forward(const std::vector<bp::Array>& xs) override {
        return { xs[0] / xs[1] };
    }
    std::vector<std::shared_ptr<Variable>> backward(const std::vector<std::shared_ptr<Variable>>& gys) override {
        auto x0 = inputs[0];
        auto x1 = inputs[1];
        auto gy = gys[0];
        auto gx0 = Variable::create(gy->data / x1->data);
        auto gx1 = Variable::create(gy->data * ( - x0->data / (x1->data * x1->data) ));
        if (!(x0->data.shape() == x1->data.shape())) {
            gx0->data = bp::sum_to(gx0->data, x0->data.shape());
            gx1->data = bp::sum_to(gx1->data, x1->data.shape());
        }
        return { gx0, gx1 };
    }
};

class Pow : public Function {
    double c;
public:
    explicit Pow(double cc) : c(cc) {}
    std::vector<bp::Array> forward(const std::vector<bp::Array>& xs) override {
        return { bp::pow(xs[0], c) };
    }
    std::vector<std::shared_ptr<Variable>> backward(const std::vector<std::shared_ptr<Variable>>& gys) override {
        auto x = inputs[0];
        auto gy = gys[0];
        auto gx = Variable::create(c * bp::pow(x->data, c - 1.0) * gy->data);
        return { gx };
    }
};

// -----------------------------
// Operator helpers: friend-like free functions
// -----------------------------
inline std::shared_ptr<Variable> add(const std::shared_ptr<Variable>& a, const std::shared_ptr<Variable>& b) {
    auto f = std::make_shared<Add>();
    auto outs = (*f)(std::vector<std::shared_ptr<Variable>>{a, b});
    return outs[0];
}
inline std::shared_ptr<Variable> mul(const std::shared_ptr<Variable>& a, const std::shared_ptr<Variable>& b) {
    auto f = std::make_shared<Mul>();
    auto outs = (*f)(std::vector<std::shared_ptr<Variable>>{a, b});
    return outs[0];
}
inline std::shared_ptr<Variable> neg(const std::shared_ptr<Variable>& a) {
    auto f = std::make_shared<Neg>();
    auto outs = (*f)(std::vector<std::shared_ptr<Variable>>{a});
    return outs[0];
}
inline std::shared_ptr<Variable> sub(const std::shared_ptr<Variable>& a, const std::shared_ptr<Variable>& b) {
    auto f = std::make_shared<Sub>();
    auto outs = (*f)(std::vector<std::shared_ptr<Variable>>{a, b});
    return outs[0];
}
inline std::shared_ptr<Variable> divv(const std::shared_ptr<Variable>& a, const std::shared_ptr<Variable>& b) {
    auto f = std::make_shared<Div>();
    auto outs = (*f)(std::vector<std::shared_ptr<Variable>>{a, b});
    return outs[0];
}
inline std::shared_ptr<Variable> powv(const std::shared_ptr<Variable>& a, double c) {
    auto f = std::make_shared<Pow>(c);
    auto outs = (*f)(std::vector<std::shared_ptr<Variable>>{a});
    return outs[0];
}

// -----------------------------
// Simple demo
// -----------------------------
inline void demo() {
    bp::Array a = bp::array({1.0, 2.0, 3.0});
    bp::Array b = bp::array({4.0, 5.0, 6.0});
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

} // namespace dz
