#pragma once

namespace bs {

    class Softmax : public Function {
    private:
        int axis_; // Softmax�� ����� �� (Attention������ ���� ������ ��)

    public:
        explicit Softmax(int axis = -1) : axis_(axis) {}

        std::vector<nb::Array> forward(const std::vector<nb::Array>& xs) override {
            const nb::Array& x = xs[0];

            // 1. ��ġ ����ȭ: �Է¿��� �ִ밪�� ���ϴ�.
            // nb::max�� axis�� ���� �ִ밪�� ���ϰ� keepdims=true�� �����Ѵٰ� ����
            nb::Array x_max = nb::max(x, axis_, true);
            nb::Array y_shifted = x - x_max; // ��ε�ĳ���� ����

            // 2. ���� �Լ� (Numerator)
            nb::Array numerator = nb::exp(y_shifted);

            // 3. �и� �ջ� (Denominator)
            // nb::sum�� axis�� ���� �ջ��ϰ� keepdims=true�� �����Ѵٰ� ����
            nb::Array denominator = nb::sum(numerator, axis_, true);

            // 4. ���� ���
            nb::Array y = numerator / denominator;

            // backward�� ���� ����� ���� (Softmax�� ����� ��Ȱ���մϴ�)
            retain_outputs_for_backward({ y });

            return { y };
        }

        // B. Backward (������) ������ �ٽ�: Chain Rule ����
        std::vector<std::shared_ptr<Variable>> backward(const std::vector<std::shared_ptr<Variable>>& gys) override {
            // y�� forward�� ����� ��Ÿ���� Variable�Դϴ�.
            auto y_ptr = get_retained_outputs()[0].lock(); // ����� ��� y�� �����ɴϴ�.
            if (!y_ptr) return { nullptr }; // ������ üũ
            auto y = y_ptr;
            auto gy = gys[0]; // dL/dy

            // 1. dL/dx = y * (gy - sum(gy * y))
            // sum(gy * y) (Reduction)
            auto sum_gy_y = sum(mul(gy, y), axis_, true); // bs::sum, bs::mul ��� ����

            // 2. gx = y * (gy - sum_gy_y)
            auto gx = mul(y, sub(gy, sum_gy_y)); // bs::sub, bs::mul ��� ����

            return { gx };
        }
    };

    // Function Wrapper (bs::softmax)
    inline std::shared_ptr<Variable> softmax(const std::shared_ptr<Variable>& x, int axis = -1) {
        auto f = std::make_shared<Softmax>(axis);
        auto outs = (*f)(std::vector<std::shared_ptr<Variable>>{x});
        return outs[0];
    }

}