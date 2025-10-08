#include <iostream>
#include <vector>

template<class T>
class Matrix {
private:
    std::vector<T> data;
    std::size_t row, col;
public:
    Matrix(std::size_t r, std::size_t c) 
        : row(r), col(c), data(r * c, T{}) {}

    T& operator()(std::size_t r, std::size_t c) { return data[r * col + c]; }
    const T& operator()(std::size_t r, std::size_t c) const { return data[r * col + c]; }

    Matrix& operator+=(const Matrix& rhs) {
        for (std::size_t i = 0; i < row; i++) {
            for (std::size_t j = 0; j < col; j++) {
                (*this)(i, j) += rhs(i, j);
            }
        }
        return *this;
    }
    friend Matrix operator+(Matrix lhs, const Matrix& rhs) {
        lhs += rhs;
        return lhs;
    }
    Matrix& operator-=(const Matrix& rhs) {
        for (std::size_t i = 0; i < row; i++) {
            for (std::size_t j = 0; j < col; j++) {
                (*this)(i, j) -= rhs(i, j);
            }
        }
        return *this;
    }
    friend Matrix operator-(Matrix lhs, const Matrix& rhs) {
        lhs -= rhs;
        return lhs;
    }
    Matrix& operator*=(const Matrix& rhs) {
        for (std::size_t i = 0; i < row; i++) {
            for (std::size_t j = 0; j < col; j++) {
                (*this)(i, j) *= rhs(i, j);
            }
        }
        return *this;
    }
    friend Matrix operator*(Matrix lhs, const Matrix& rhs) {
        lhs *= rhs;
        return lhs;
    }
    Matrix& operator/=(const Matrix& rhs) {
        for (std::size_t i = 0; i < row; i++) {
            for (std::size_t j = 0; j < col; j++) {
                (*this)(i, j) /= rhs(i, j);
            }
        }
        return *this;
    }
    friend Matrix operator/(Matrix lhs, const Matrix& rhs) {
        lhs /= rhs;
        return lhs;
    }
};

int main() {
    Matrix<double> m1(10000, 10000);
    Matrix<double> m2(10000, 10000);

    m1(2, 1) = 50.0;
    m2(2, 1) = 40.0;
    m1 = m1 + m2;

    std::cout << m1(2, 1) << std::endl;

    return 0;
}