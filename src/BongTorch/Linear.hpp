#pragma once

// #include "bongpy.hpp" ����: nb::Array, nb::ones_like, nb::sum_to, nb::reshape, nb::transpose, nb::pow, nb::as_array
// #include "Core.hpp"
#include <memory> // �ӽ÷� ����

class Linear : public Module {
private:

public:
	std::shared_ptr<Parameter> W; // ����ġ
	std::shared_ptr<Parameter> b; // ���� �ټ�

	Linear(int in_features, int out_features) {
		W = Parameter::create(nb::rand({ in_features, out_features })); // bongpy�� rand �ʿ�
		b = Parameter::create(nb::zeros({ out_features })); // bongpy�� zero �ʿ�
	}

	std::shared_ptr<Variable> forward(const std::shared_ptr<Variable>& x) {
		auto y_matmul = matmul(x, W);
		auto y = add(y_matmul, b);
		return y;
	}
}