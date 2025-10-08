#ifndef MODULE_HPP
#define MODULE_HPP

#include "core.hpp"
#include <map>
#include <vector>
#include <memory>
#include "../NumBong.hpp"

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

#endif // MODULE_HPP