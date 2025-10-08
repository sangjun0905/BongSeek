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
using TensorData = nb::Tensor<TensorValueType, TensorRank>;
using Shape = nb::Shape;
using TensorShape = typename TensorData::shape_type;

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

class Function;

class Variable : public std::enable_shared_from_this<Variable> {
public:
    TensorData data;
    std::string name;
    std::shared_ptr<Variable> grad;
    std::shared_ptr<Function> creator;
    int generation = 0;

    explicit Variable(const TensorData& arr, const std::string& n = "") : data(arr), name(n) {}

    static std::shared_ptr<Variable> create(const TensorData& arr, const std::string& n = "") {
        return std::make_shared<Variable>(arr, n);
    }

    Shape shape() const { return data.shape_vector(); }
    int ndim() const { return static_cast<int>(data.ndim()); }
    size_t size() const { return data.size(); }
    auto dtype() const { return data.dtype(); }

    void set_creator(const std::shared_ptr<Function>& f);
    void unchain() { creator.reset(); }
    void cleargrad() { grad.reset(); }

    void backward(bool retain_grad=false, bool create_graph=false);
    void unchain_backward();

    void to_cpu() { data = nb::as_cpu(data); }
    void to_gpu() { data = nb::as_gpu(data); }

    void print(const std::string& prefix="") const {
        std::cout << prefix << "Variable(name=" << name << ", shape=" << data.shape_string() << ")\n";
    }
};

class Function : public std::enable_shared_from_this<Function> {
public:
    std::vector<std::shared_ptr<Variable>> inputs;
    std::vector<std::weak_ptr<Variable>> outputs;
    int generation = 0;

    virtual ~Function() = default;

    std::vector<std::shared_ptr<Variable>> operator()(const std::vector<std::shared_ptr<Variable>>& in_vars) {
        inputs = in_vars;

        std::vector<TensorData> xs;
        xs.reserve(inputs.size());
        for (const auto& v : inputs) {
            xs.push_back(v->data);
        }

        std::vector<TensorData> ys = forward(xs);

        std::vector<std::shared_ptr<Variable>> out_vars;
        out_vars.reserve(ys.size());
        outputs.clear();
        for (auto& y : ys) {
            auto out = Variable::create(nb::as_array(y));
            out_vars.push_back(out);
        }

        if (Config::enable_backprop) {
            generation = 0;
            for (const auto& v : inputs) {
                generation = std::max(generation, v->generation);
            }
            for (auto& out : out_vars) {
                out->set_creator(this->shared_from_this());
                outputs.push_back(out);
            }
        }

        return out_vars;
    }

    virtual std::vector<TensorData> forward(const std::vector<TensorData>& xs) = 0;
    virtual std::vector<std::shared_ptr<Variable>> backward(const std::vector<std::shared_ptr<Variable>>& gys) = 0;
};

inline void Variable::set_creator(const std::shared_ptr<Function>& f) {
    creator = f;
    generation = f->generation + 1;
}

inline void Variable::backward(bool retain_grad, bool create_graph) {
    if (!grad) {
        grad = Variable::create(nb::ones_like(data));
    }

    std::vector<std::shared_ptr<Function>> funcs;
    std::set<Function*> seen;

    auto add_func = [&](const std::shared_ptr<Function>& f) {
        if (!f) return;
        if (seen.insert(f.get()).second) {
            funcs.push_back(f);
            std::sort(funcs.begin(), funcs.end(), [](const std::shared_ptr<Function>& a, const std::shared_ptr<Function>& b) {
                return a->generation < b->generation;
            });
        }
    };

    add_func(creator);

    while (!funcs.empty()) {
        auto f = funcs.back();
        funcs.pop_back();

        std::vector<std::shared_ptr<Variable>> gys;
        gys.reserve(f->outputs.size());
        for (auto& w : f->outputs) {
            if (auto outp = w.lock()) gys.push_back(outp->grad);
            else gys.push_back(nullptr);
        }

        {
            UsingConfig tmp(Config::enable_backprop, create_graph);
            auto gxs = f->backward(gys);

            for (size_t i = 0; i < f->inputs.size(); ++i) {
                auto x = f->inputs[i];
                auto gx = (i < gxs.size() ? gxs[i] : nullptr);
                if (!gx) continue;

                if (!x->grad) x->grad = gx;
                else x->grad->data.iadd(gx->data);

                if (x->creator) add_func(x->creator);
            }
        }

        if (!retain_grad) {
            for (auto& w : f->outputs) {
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
        auto f = funcs.back();
        funcs.pop_back();
        for (auto& x : f->inputs) {
            if (x->creator) {
                funcs.push_back(x->creator);
                x->unchain();
            }
        }
    }
}

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
        return { xs[0] + xs[1] };
    }
};

class Mul : public Function {
public:
    std::vector<TensorData> forward(const std::vector<TensorData>& xs) override {
        return { xs[0] * xs[1] };
    }

    std::vector<std::shared_ptr<Variable>> backward(const std::vector<std::shared_ptr<Variable>>& gys) override {
        auto x0 = inputs[0];
        auto x1 = inputs[1];
        auto gy = gys[0];
        auto gx0 = Variable::create(gy->data * x1->data);
        auto gx1 = Variable::create(gy->data * x0->data);
        if (!(x0->data.getShape() == x1->data.getShape())) {
            gx0->data = nb::sum_to(gx0->data, x0->data.getShape());
            gx1->data = nb::sum_to(gx1->data, x1->data.getShape());
        }
        return { gx0, gx1 };
    }
};

class Neg : public Function {
public:
    std::vector<TensorData> forward(const std::vector<TensorData>& xs) override {
        return { -xs[0] };
    }

    std::vector<std::shared_ptr<Variable>> backward(const std::vector<std::shared_ptr<Variable>>& gys) override {
        auto gy = gys[0];
        return { Variable::create(-gy->data) };
    }
};

class Sub : public Function {
    TensorShape x0_shape{}, x1_shape{};
public:
    std::vector<TensorData> forward(const std::vector<TensorData>& xs) override {
        x0_shape = xs[0].getShape();
        x1_shape = xs[1].getShape();
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
    std::vector<TensorData> forward(const std::vector<TensorData>& xs) override {
        return { xs[0] / xs[1] };
    }

    std::vector<std::shared_ptr<Variable>> backward(const std::vector<std::shared_ptr<Variable>>& gys) override {
        auto x0 = inputs[0];
        auto x1 = inputs[1];
        auto gy = gys[0];
        auto gx0 = Variable::create(gy->data / x1->data);
        auto gx1 = Variable::create(gy->data * (-x0->data / (x1->data * x1->data)));
        if (!(x0->data.getShape() == x1->data.getShape())) {
            gx0->data = nb::sum_to(gx0->data, x0->data.getShape());
            gx1->data = nb::sum_to(gx1->data, x1->data.getShape());
        }
        return { gx0, gx1 };
    }
};

class Pow : public Function {
    double c;
public:
    explicit Pow(double cc) : c(cc) {}

    std::vector<TensorData> forward(const std::vector<TensorData>& xs) override {
        return { nb::pow(xs[0], c) };
    }

    std::vector<std::shared_ptr<Variable>> backward(const std::vector<std::shared_ptr<Variable>>& gys) override {
        auto x = inputs[0];
        auto gy = gys[0];
        auto gx = Variable::create(c * nb::pow(x->data, c - 1.0) * gy->data);
        return { gx };
    }
};

inline std::shared_ptr<Variable> add(const std::shared_ptr<Variable>& a, const std::shared_ptr<Variable>& b) {
    auto f = std::make_shared<Add>();
    auto outs = (*f)(std::vector<std::shared_ptr<Variable>>{a, b});
    return outs[0];
}

inline std::shared_ptr<Variable> operator+(const std::shared_ptr<Variable>& a, const std::shared_ptr<Variable>& b) {
    return add(a, b);
}

inline std::shared_ptr<Variable> mul(const std::shared_ptr<Variable>& a, const std::shared_ptr<Variable>& b) {
    auto f = std::make_shared<Mul>();
    auto outs = (*f)(std::vector<std::shared_ptr<Variable>>{a, b});
    return outs[0];
}

inline std::shared_ptr<Variable> operator*(const std::shared_ptr<Variable>& a, const std::shared_ptr<Variable>& b) {
    return mul(a, b);
}

inline std::shared_ptr<Variable> neg(const std::shared_ptr<Variable>& a) {
    auto f = std::make_shared<Neg>();
    auto outs = (*f)(std::vector<std::shared_ptr<Variable>>{a});
    return outs[0];
}

inline std::shared_ptr<Variable> operator-(const std::shared_ptr<Variable>& a) {
    return neg(a);
}

inline std::shared_ptr<Variable> sub(const std::shared_ptr<Variable>& a, const std::shared_ptr<Variable>& b) {
    auto f = std::make_shared<Sub>();
    auto outs = (*f)(std::vector<std::shared_ptr<Variable>>{a, b});
    return outs[0];
}

inline std::shared_ptr<Variable> operator-(const std::shared_ptr<Variable>& a, const std::shared_ptr<Variable>& b) {
    return sub(a, b);
}

inline std::shared_ptr<Variable> divv(const std::shared_ptr<Variable>& a, const std::shared_ptr<Variable>& b) {
    auto f = std::make_shared<Div>();
    auto outs = (*f)(std::vector<std::shared_ptr<Variable>>{a, b});
    return outs[0];
}

inline std::shared_ptr<Variable> operator/(const std::shared_ptr<Variable>& a, const std::shared_ptr<Variable>& b) {
    return divv(a, b);
}

inline std::shared_ptr<Variable> powv(const std::shared_ptr<Variable>& a, double c) {
    auto f = std::make_shared<Pow>(c);
    auto outs = (*f)(std::vector<std::shared_ptr<Variable>>{a});
    return outs[0];
}

inline std::shared_ptr<Variable> operator^(const std::shared_ptr<Variable>& a, double c) {
    return powv(a, c);
}

inline void demo() {
    TensorData a(3, 3, 3);
    TensorData b(3, 3, 3);

    auto A = Variable::create(a, "A");
    auto B = Variable::create(b, "B");

    auto C = add(mul(A, B), A);
    C->print("C: ");
    C->backward();
    if (A->grad) {
        std::cout << "A.grad shape: " << A->grad->data.shape_string() << "\n";
    } else {
        std::cout << "A.grad none\n";
    }
}

#endif
