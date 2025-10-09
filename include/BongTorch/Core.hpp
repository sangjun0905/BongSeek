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
#include <map>
#include "../NumBong/Tensor.hpp"
#include "../NumBong/BFloat16.hpp"

using TensorValueType = nb::BFloat16;
constexpr std::size_t TensorRank = 3;
using Tensor = nb::Tensor<TensorValueType, TensorRank>;
using Shape = nb::Shape;
using TensorShape = typename Tensor::shape_type;

namespace bs {

class Variable : public std::enable_shared_from_this<Variable> {
public:
    Tensor data;
    std::string name;
    explicit Variable(const Tensor& arr, const std::string& n = "") : data(arr), name(n) {}
    static std::shared_ptr<Variable> create(const Tensor& arr, const std::string& n = "") {
        return std::make_shared<Variable>(arr, n);
    }
    Shape shape() const { return data.shape_vector(); }
    int ndim() const { return static_cast<int>(data.ndim()); }
    size_t size() const { return data.size(); }
    auto dtype() const { return data.dtype(); }
};

class Parameter : public Variable {
public:
    explicit Parameter(const Tensor& arr, const std::string& n = "") : Variable(arr, n) {}
    static std::shared_ptr<Parameter> create(const Tensor& arr, const std::string& n = "") {
        return std::make_shared<Parameter>(arr, n);
    }
};

class Function : public std::enable_shared_from_this<Function> {
public:
    virtual ~Function() = default;
    std::shared_ptr<Variable> operator()(const std::vector<std::shared_ptr<Variable>>& in_vars) {
        std::vector<Tensor> xs;   
        xs.reserve(in_vars.size());
        for (const auto& v : in_vars) {
            xs.push_back(v->data);
        }
        std::vector<Tensor> ys = this->forward(xs);
        auto out = Variable::create(nb::as_array(ys[0]));
        return out;
        }
    virtual std::vector<Tensor> forward(const std::vector<Tensor>& xs) = 0;
};

class Module : public std::enable_shared_from_this<Module> {
private:
    std::map<std::string, std::shared_ptr<Parameter>> parameters;
    std::map<std::string, std::shared_ptr<Module>> children;
public:
    virtual ~Module() = default;
    virtual std::shared_ptr<Variable> forward(const std::shared_ptr<Variable>& x) = 0;
    std::shared_ptr<Variable> operator()(const std::shared_ptr<Variable>& x) {
        return this->forward(x);
}
    void register_parameter(const std::string& name, const std::shared_ptr<Parameter>& param) {
        parameters[name] = param;
    }
    void add_module(const std::string& name, const std::shared_ptr<Module>& module) {
        children[name] = module;
    }
    std::vector<std::shared_ptr<Parameter>> get_parameters() {
        std::vector<std::shared_ptr<Parameter>> all_params;
        for (const auto& pair : parameters) {
            all_params.push_back(pair.second);
            }
        for (const auto& pair : children) {
            auto child_params = pair.second->get_parameters();
            all_params.insert(all_params.end(), child_params.begin(), child_params.end());
        }
        return all_params;
    }
};

// --- 추론 전용 Function 구현 ---
class Add : public Function {
public:
    std::vector<Tensor> forward(const std::vector<Tensor>& xs) override {
        return { xs[0] + xs[1] };
    }
};

class Mul : public Function {
public:
    std::vector<Tensor> forward(const std::vector<Tensor>& xs) override {
        return { xs[0] * xs[1] };
    }
};

class Neg : public Function {
public:
    std::vector<Tensor> forward(const std::vector<Tensor>& xs) override {
        const auto& input = xs[0];
        auto output = Tensor(input.getShape());
        const auto* src = input.data();
        auto* dst = output.data();
        for (size_t i = 0; i < input.size(); ++i) {
            dst[i] = nb::BFloat16(-static_cast<float>(src[i]));
        }
        return { output };
    }
};

class Sub : public Function {
public:
    std::vector<Tensor> forward(const std::vector<Tensor>& xs) override {
        return { xs[0] - xs[1] };
    }
};

class Div : public Function {
public:
    std::vector<Tensor> forward(const std::vector<Tensor>& xs) override {
        return { xs[0] / xs[1] };
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

// --- 연산자 오버로딩 (Operator Overloading) ---

inline std::shared_ptr<Variable> apply_op(const std::shared_ptr<Variable>& a, const std::shared_ptr<Variable>& b, const std::shared_ptr<Function>& f) {
    return (*f)({a, b});
}

inline std::shared_ptr<Variable> apply_op(const std::shared_ptr<Variable>& a, const std::shared_ptr<Function>& f) {
    return (*f)({a});
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
    return (*f)({a});
}

} // namespace bs

#endif