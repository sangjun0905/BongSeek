#pragma once

// #include "bongpy.hpp" 가정: nb::Array, nb::ones_like, nb::sum_to, nb::reshape, nb::transpose, nb::pow, nb::as_array
// #include "Core.hpp"
#include <memory> // 임시로 선언

class Linear : public Module {
private:

public:
	std::shared_ptr<Parameter> W; // 가중치
	std::shared_ptr<Parameter> b; // 편향 텐서

	Linear(int in_features, int out_features) {
		W = Parameter::create(nb::rand({ in_features, out_features })); // bongpy에 rand 필요
		b = Parameter::create(nb::zeros({ out_features })); // bongpy에 zero 필요
	}

	std::shared_ptr<Variable> forward(const std::shared_ptr<Variable>& x) {
		auto y_matmul = matmul(x, W);
		auto y = add(y_matmul, b);
		return y;
	}
}