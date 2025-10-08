#pragma once

namespace bs {

    class Softmax : public Function {
    private:
        int axis_; // Softmax가 적용될 축 (Attention에서는 보통 마지막 축)

    public:
        explicit Softmax(int axis = -1) : axis_(axis) {}

        std::vector<nb::Array> forward(const std::vector<nb::Array>& xs) override {
            const nb::Array& x = xs[0];

            // 1. 수치 안정화: 입력에서 최대값을 뺍니다.
            // nb::max는 axis를 따라 최대값을 구하고 keepdims=true로 설정한다고 가정
            nb::Array x_max = nb::max(x, axis_, true);
            nb::Array y_shifted = x - x_max; // 브로드캐스팅 빼기

            // 2. 지수 함수 (Numerator)
            nb::Array numerator = nb::exp(y_shifted);

            // 3. 분모 합산 (Denominator)
            // nb::sum은 axis를 따라 합산하고 keepdims=true로 설정한다고 가정
            nb::Array denominator = nb::sum(numerator, axis_, true);

            // 4. 최종 결과
            nb::Array y = numerator / denominator;

            // backward를 위해 출력을 저장 (Softmax는 출력을 재활용합니다)
            retain_outputs_for_backward({ y });

            return { y };
        }

        // B. Backward (역전파) 구현의 핵심: Chain Rule 적용
        std::vector<std::shared_ptr<Variable>> backward(const std::vector<std::shared_ptr<Variable>>& gys) override {
            // y는 forward의 출력을 나타내는 Variable입니다.
            auto y_ptr = get_retained_outputs()[0].lock(); // 저장된 출력 y를 가져옵니다.
            if (!y_ptr) return { nullptr }; // 안전성 체크
            auto y = y_ptr;
            auto gy = gys[0]; // dL/dy

            // 1. dL/dx = y * (gy - sum(gy * y))
            // sum(gy * y) (Reduction)
            auto sum_gy_y = sum(mul(gy, y), axis_, true); // bs::sum, bs::mul 사용 가정

            // 2. gx = y * (gy - sum_gy_y)
            auto gx = mul(y, sub(gy, sum_gy_y)); // bs::sub, bs::mul 사용 가정

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