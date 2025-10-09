// PositionEncoding Function (dz_functions.hpp)

class PositionEncoding : public Function {
private:
    int max_len; // 최대 시퀀스 길이
    int d_model; // 모델 차원 (d_model)
    // 미리 계산된 PE 행렬을 저장할 수 있습니다 (선택 사항이지만 효율적)
    // nb::Array pe_array; 

public:
    // 생성자: PE 행렬을 계산하는 데 필요한 정보를 받습니다.
    PositionEncoding(int max_len, int d_model) : max_len(max_len), d_model(d_model) {
        // 실제 구현에서는 여기서 PE 행렬을 미리 계산하여 pe_array에 저장할 수 있습니다.
    }

    // P 텐서를 생성하는 보조 함수 (봉파이 기능 사용 가정)
    nb::Array create_pe_array(const nb::Array& input_data) {
        // 실제 트랜스포머 입력 (Sequence Length)에 맞게 PE를 자르거나 생성해야 함
        int S = input_data.shape()[1]; // 입력 텐서의 시퀀스 길이

        // C++에서 PE 행렬을 생성하는 로직 (수학 공식 적용)

        nb::Array P = nb::zeros({ S, d_model }); // [S, D] 크기의 텐서 생성

        // ... C++ 반복문/봉파이 벡터 연산을 사용하여 PE 공식(sin/cos) 구현 ...

        // 예시: C++의 sin/cos 함수와 nb::pow를 사용하여 P 채우기
        for (int pos = 0; pos < S; ++pos) {
            for (int i = 0; i < d_model; ++i) {
                // 주파수 계산: 1 / (10000^(2i/d_model))
                double freq = 1.0 / std::pow(10000.0, (double)i / d_model);

                if (i % 2 == 0) { // 짝수 인덱스는 sin
                    P.set_value({ pos, i }, std::sin(pos * freq));
                }
                else { // 홀수 인덱스는 cos
                    P.set_value({ pos, i }, std::cos(pos * freq));
                }
            }
        }

        return P;
    }

    std::vector<nb::Array> forward(const std::vector<nb::Array>& xs) override {
        // xs[0]은 Input Embedding 텐서입니다.
        nb::Array X = xs[0];
        nb::Array P = create_pe_array(X); // 입력 크기에 맞는 PE 텐서 생성

        // PE는 학습 가능한 파라미터가 아니므로, 단순히 원소별 덧셈을 수행합니다.
        nb::Array Y = X + P;

        return { Y };
    }

    std::vector<std::shared_ptr<Variable>> backward(const std::vector<std::shared_ptr<Variable>>& gys) override {
        // PE는 학습 가능한 파라미터가 없으므로, 기울기는 그대로 통과합니다.
        auto gy = gys[0];
        // Y = X + P 이므로, dL/dX = dL/dY 입니다.
        return { Variable::create(gy->data) };
    }
};

// Function Wrapper
inline std::shared_ptr<Variable> position_encoding(const std::shared_ptr<Variable>& x, int max_len, int d_model) {
    auto f = std::make_shared<PositionEncoding>(max_len, d_model);
    auto outs = (*f)(std::vector<std::shared_ptr<Variable>>{x});
    return outs[0];
}