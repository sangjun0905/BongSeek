#pragma once

#include "Core.hpp"

class Linear : public bs::Function {
public:
	std::shared_ptr<Parameter> W; // 가중치
	std::shared_ptr<Parameter> b; // 편향 텐서

	Linear(int in_features, int out_features) {
		W = Parameter::create(nb::rand({ in_features, out_features })); // Numbong에서 rand
		b = Parameter::create(nb::zeros({ out_features })); // Numbong에서 zero
	}

	std::shared_ptr<Variable> forward(const std::shared_ptr<Variable>& x) {
		auto y_matmul = matmul(x, W);
		auto y = add(y_matmul, b);
		return y;
	}
};