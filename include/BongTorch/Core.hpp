#ifndef CORE_HPP
#define CORE_HPP

#include <algorithm>
#include <array>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include "../NumBong/Tensor.hpp" 

/*
struct Metadatainfo {
	size_t offset_start;
	size_t offset_end;
	nb::Shape shape;
	std::string dtype;
};*/

struct TensorInfo {
    std::string dtype;
    std::vector<size_t> shape;
    size_t offset_start;
    size_t offset_end;
};

typedef TensorInfo Metadatainfo;

using MetadataMap = std::unordered_map<std::string, TensorInfo>;

using TensorValueType = nb::BFloat16;
constexpr std::size_t TensorRank = 3;
<<<<<<< HEAD
using Tensor = nb::Tensor<TensorValueType, TensorRank>;  //
using Shape = nb::Shape;                                     // ->Shape이라고 별칭 지정(자료형)
using TensorShape = typename Tensor::shape_type;         // ->Tensor내부에서 공식적으로 사용하는  shape타입 별칭

// 역전파 관련 Config, UsingConfig 등 모두 제거됨.

class Function;
=======
using Tensor = nb::Tensor<TensorValueType, TensorRank>;
using Shape = nb::Shape;
using TensorShape = typename Tensor::shape_type;

namespace bs {
>>>>>>> origin/BongTorchJW

class Variable : public std::enable_shared_from_this<Variable> {
public:
    Tensor data;
    std::string name;
<<<<<<< HEAD
    // 역전파 관련 멤버 제거: grad, creator, generation

    explicit Variable(const Tensor& arr, const std::string& n = "") : data(arr), name(n) {}

    static std::shared_ptr<Variable> create(const Tensor& arr, const std::string& n = "") {
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
=======
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
>>>>>>> origin/BongTorchJW
    std::shared_ptr<Variable> operator()(const std::vector<std::shared_ptr<Variable>>& in_vars) {
        std::vector<Tensor> xs;   
        xs.reserve(in_vars.size());
        for (const auto& v : in_vars) {
<<<<<<< HEAD
            xs.push_back(v->data);         //_> Variable의 텐서만 뽑아서 담음
        }

        std::vector<Tensor> ys = forward(xs);
        
        // Function은 오직 하나의 Variable을 출력한다고 가정
        auto out = Variable::create(nb::as_array(ys[0]));
        
        return out;   //-> 최종 산출물은 Variable 포인터
            }

    // forward는 TensorData 기반으로 오버라이딩합니다.
    virtual std::vector<Tensor> forward(const std::vector<Tensor>& xs) = 0;
    
};

class Parameter : public Variable {
public:
    explicit Parameter(const Tensor& arr, const std::string& n = "") : Variable(arr, n) {}

    static std::shared_ptr<Parameter> create(const Tensor& arr, const std::string& n = "") {
        return std::make_shared<Parameter>(arr, n);
=======
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
>>>>>>> origin/BongTorchJW
    }
};

// --- 추론 전용 Function 구현 ---
class Add : public Function {
public:
    std::vector<Tensor> forward(const std::vector<Tensor>& xs) override {
<<<<<<< HEAD
        return { xs[0].add( xs[1]) };
=======
        return { xs[0] + xs[1] };
>>>>>>> origin/BongTorchJW
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
<<<<<<< HEAD
        return { xs[0].neg() };
=======
        const auto& input = xs[0];
        auto output = Tensor(input.getShape());
        const auto* src = input.data();
        auto* dst = output.data();
        for (size_t i = 0; i < input.size(); ++i) {
            dst[i] = nb::BFloat16(-static_cast<float>(src[i]));
        }
        return { output };
>>>>>>> origin/BongTorchJW
    }
};

class Sub : public Function {
public:
    std::vector<Tensor> forward(const std::vector<Tensor>& xs) override {
<<<<<<< HEAD
        return { xs[0].sub(xs[1]) };
=======
        return { xs[0] - xs[1] };
>>>>>>> origin/BongTorchJW
    }
};

class Div : public Function {
public:
    std::vector<Tensor> forward(const std::vector<Tensor>& xs) override {
<<<<<<< HEAD
        return { xs[0].div(xs[1]) };
        
=======
        return { xs[0] / xs[1] };
>>>>>>> origin/BongTorchJW
    }
};

class Pow : public Function {
    double c;
public:
    explicit Pow(double cc) : c(cc) {}

    std::vector<Tensor> forward(const std::vector<Tensor>& xs) override {
        return { nb::pow(xs[0], c) };
<<<<<<< HEAD
    }                           //---> Function을 상속받으므로 Function의 forward오버라이딩 실제 nb::pow()연산 수행
=======
    }
>>>>>>> origin/BongTorchJW
};

// --- 연산자 오버로딩 (Operator Overloading) ---

inline std::shared_ptr<Variable> apply_op(const std::shared_ptr<Variable>& a, const std::shared_ptr<Variable>& b, const std::shared_ptr<Function>& f) {
<<<<<<< HEAD
    auto out = (*f)({a, b});  //-> <add>function객체 생성 후 a,b 넘김
    return out;
}

inline std::shared_ptr<Variable> apply_op(const std::shared_ptr<Variable>& a, const std::shared_ptr<Function>& f) {
    auto out = (*f)({a});
    return out;
=======
    return (*f)({a, b});
}

inline std::shared_ptr<Variable> apply_op(const std::shared_ptr<Variable>& a, const std::shared_ptr<Function>& f) {
    return (*f)({a});
}

inline std::shared_ptr<Variable> add(const std::shared_ptr<Variable>& a, const std::shared_ptr<Variable>& b) {
    return apply_op(a, b, std::make_shared<Add>());
>>>>>>> origin/BongTorchJW
}

inline std::shared_ptr<Variable> add(const std::shared_ptr<Variable>& a, const std::shared_ptr<Variable>& b) {
    return apply_op(a, b, std::make_shared<Add>()); //->2. apply_op()호출
}


inline std::shared_ptr<Variable> mul(const std::shared_ptr<Variable>& a, const std::shared_ptr<Variable>& b) {
    return apply_op(a, b, std::make_shared<Mul>());
}
<<<<<<< HEAD

=======
>>>>>>> origin/BongTorchJW

inline std::shared_ptr<Variable> neg(const std::shared_ptr<Variable>& a) {
    return apply_op(a, std::make_shared<Neg>());
}
<<<<<<< HEAD

=======
>>>>>>> origin/BongTorchJW

inline std::shared_ptr<Variable> sub(const std::shared_ptr<Variable>& a, const std::shared_ptr<Variable>& b) {
    return apply_op(a, b, std::make_shared<Sub>());
}
<<<<<<< HEAD

=======
>>>>>>> origin/BongTorchJW

inline std::shared_ptr<Variable> divv(const std::shared_ptr<Variable>& a, const std::shared_ptr<Variable>& b) {
    return apply_op(a, b, std::make_shared<Div>());
}

inline std::shared_ptr<Variable> powv(const std::shared_ptr<Variable>& a, double c) {
    auto f = std::make_shared<Pow>(c);
<<<<<<< HEAD
    auto outs = (*f)(std::vector<std::shared_ptr<Variable>>{a});
    return outs;
}

=======
    return (*f)({a});
}

} // namespace bs

>>>>>>> origin/BongTorchJW
#endif