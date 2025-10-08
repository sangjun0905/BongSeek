// PositionEncoding Function (dz_functions.hpp)

class PositionEncoding : public Function {
private:
    int max_len; // �ִ� ������ ����
    int d_model; // �� ���� (d_model)
    // �̸� ���� PE ����� ������ �� �ֽ��ϴ� (���� ���������� ȿ����)
    // nb::Array pe_array; 

public:
    // ������: PE ����� ����ϴ� �� �ʿ��� ������ �޽��ϴ�.
    PositionEncoding(int max_len, int d_model) : max_len(max_len), d_model(d_model) {
        // ���� ���������� ���⼭ PE ����� �̸� ����Ͽ� pe_array�� ������ �� �ֽ��ϴ�.
    }

    // P �ټ��� �����ϴ� ���� �Լ� (������ ��� ��� ����)
    nb::Array create_pe_array(const nb::Array& input_data) {
        // ���� Ʈ�������� �Է� (Sequence Length)�� �°� PE�� �ڸ��ų� �����ؾ� ��
        int S = input_data.shape()[1]; // �Է� �ټ��� ������ ����

        // C++���� PE ����� �����ϴ� ���� (���� ���� ����)

        nb::Array P = nb::zeros({ S, d_model }); // [S, D] ũ���� �ټ� ����

        // ... C++ �ݺ���/������ ���� ������ ����Ͽ� PE ����(sin/cos) ���� ...

        // ����: C++�� sin/cos �Լ��� nb::pow�� ����Ͽ� P ä���
        for (int pos = 0; pos < S; ++pos) {
            for (int i = 0; i < d_model; ++i) {
                // ���ļ� ���: 1 / (10000^(2i/d_model))
                double freq = 1.0 / std::pow(10000.0, (double)i / d_model);

                if (i % 2 == 0) { // ¦�� �ε����� sin
                    P.set_value({ pos, i }, std::sin(pos * freq));
                }
                else { // Ȧ�� �ε����� cos
                    P.set_value({ pos, i }, std::cos(pos * freq));
                }
            }
        }

        return P;
    }

    std::vector<nb::Array> forward(const std::vector<nb::Array>& xs) override {
        // xs[0]�� Input Embedding �ټ��Դϴ�.
        nb::Array X = xs[0];
        nb::Array P = create_pe_array(X); // �Է� ũ�⿡ �´� PE �ټ� ����

        // PE�� �н� ������ �Ķ���Ͱ� �ƴϹǷ�, �ܼ��� ���Һ� ������ �����մϴ�.
        nb::Array Y = X + P;

        return { Y };
    }

    std::vector<std::shared_ptr<Variable>> backward(const std::vector<std::shared_ptr<Variable>>& gys) override {
        // PE�� �н� ������ �Ķ���Ͱ� �����Ƿ�, ����� �״�� ����մϴ�.
        auto gy = gys[0];
        // Y = X + P �̹Ƿ�, dL/dX = dL/dY �Դϴ�.
        return { Variable::create(gy->data) };
    }
};

// Function Wrapper
inline std::shared_ptr<Variable> position_encoding(const std::shared_ptr<Variable>& x, int max_len, int d_model) {
    auto f = std::make_shared<PositionEncoding>(max_len, d_model);
    auto outs = (*f)(std::vector<std::shared_ptr<Variable>>{x});
    return outs[0];
}