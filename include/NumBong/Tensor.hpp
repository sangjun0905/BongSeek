#pragma once

#include <iostream>
#include <vector>
#include <array>
#include <cassert>
#include <type_traits>
#include <memory>
#include <numeric>
#include <algorithm>
#include <string>
#include <cstdlib> // For rand()

namespace nb {

// Use std::array for shape to be consistent with Tensor class
template<std::size_t Rank>
using Shape = std::array<std::size_t, Rank>;

template<class T, std::size_t Rank>
class Tensor {
private:
    std::shared_ptr<T[]> storage;
    T* base = nullptr;
    std::ptrdiff_t offset = 0;
    std::array<std::size_t, Rank> shape_{};
    std::array<std::ptrdiff_t, Rank> stride_{};

    Tensor(
        std::shared_ptr<T[]> st, 
        T* base, 
        std::ptrdiff_t off, 
        std::array<std::size_t, Rank> sh, 
        std::array<std::ptrdiff_t, Rank> stv)
        : storage(std::move(st)), base(base), offset(off), shape_(sh), stride_(stv) {}
    
public:
    using shape_type = std::array<std::size_t, Rank>;

    Tensor() : shape_{}, stride_{} {
        // Default constructor creates an empty tensor
    }

    Tensor(const std::array<std::size_t, Rank>& shp) : shape_(shp) {
        std::size_t total = 1; for (auto d : shape_) total *= d;
        storage = std::make_shared<T[]>(total);
        base = storage.get();
        stride_[Rank-1] = 1;
        for (std::size_t i = Rank - 1; i > 0; i--) {
            stride_[i - 1] = stride_[i] * shape_[i];
        }
    }

    template<class... Ds>
    explicit Tensor(Ds... dims)
    requires (sizeof...(Ds) == Rank && (std::conjunction_v<std::is_integral<std::decay_t<Ds>>...>))
    : shape_{ static_cast<std::size_t>(dims)... } {
        storage = std::make_shared<T[]>( (std::size_t{1} * ... * static_cast<std::size_t>(dims)) );
        base = storage.get();
        stride_[Rank - 1] = 1; // Last element is 1.
        for (std::size_t i = Rank - 1; i > 0; i--) {
            stride_[i - 1] = stride_[i] * static_cast<std::ptrdiff_t>(shape_[i]);
        }
    }

    T* data() { return base + offset; }
    const T* data() const { return base + offset; }
    
    const auto& getShape() const { return shape_; }
    const auto& getStride() const { return stride_; }
    std::vector<std::size_t> shape_vector() const { return std::vector<std::size_t>(shape_.begin(), shape_.end()); }
    std::size_t ndim() const { return Rank; }
    const char* dtype() const { return typeid(T).name(); }

    std::string shape_string() const {
        std::string s = "(";
        for(size_t i = 0; i < shape_.size(); ++i) {
            s += std::to_string(shape_[i]);
            if (i < shape_.size() - 1) s += ", ";
        }
        s += ")";
        return s;
    }
    
    void iadd(const Tensor& other) {
        // Dummy implementation
        for (size_t i = 0; i < size(); ++i) {
            // This is not a correct broadcasted add
            // data()[i] += other.data()[i];
        }
    }

    template<class... Ix>
    T& operator()(Ix... ix) {
        std::array<std::ptrdiff_t, Rank> idx{ static_cast<std::ptrdiff_t>(ix)... };
        std::ptrdiff_t off = offset;
        for (std::size_t i = 0; i < Rank; i++) {
            off += idx[i] * stride_[i];
        }
        return *(base + off); 
    }
    template<class... Ix>
    const T& operator()(Ix... ix) const { 
        return const_cast<Tensor&>(*this)(ix...); 
    }

    Tensor transpose() const {
        auto sh = shape_;
        auto st = stride_;
        std::swap(sh[Rank-2], sh[Rank-1]);
        std::swap(st[Rank-2], st[Rank-1]);
        return Tensor(storage, base, 0, sh, st);
    }
    Tensor transpose(std::size_t a, std::size_t b) const {
        auto sh = shape_;
        auto st = stride_;
        std::swap(sh[a], sh[b]);
        std::swap(st[a], st[b]);
        return Tensor(storage, base, 0, sh, st);
    }

    void fill(T val) { if(storage) std::fill(data(), data() + size(), val); }
    size_t size() const { 
        if (!storage) return 0;
        size_t s = 1;
        for(auto d : shape_) s *= d;
        return s; 
    }

    static void gemm2d_naive(const T* A, std::ptrdiff_t As0, std::ptrdiff_t As1,
                            const T* B, std::ptrdiff_t Bs0, std::ptrdiff_t Bs1,
                            T* C, std::ptrdiff_t Cs0, std::ptrdiff_t Cs1,
                            std::size_t M, std::size_t K, std::size_t N) {
        for (std::size_t i = 0; i < M; i++) {
            for (std::size_t j = 0; j < N; j++) {
                T acc = T{};
                for (std::size_t k = 0; k < K; k++) {
                    const T a = *(A + static_cast<std::ptrdiff_t>(i)*As0 + static_cast<std::ptrdiff_t>(k)*As1);
                    const T b = *(B + static_cast<std::ptrdiff_t>(k)*Bs0 + static_cast<std::ptrdiff_t>(j)*Bs1);
                    acc += a * b;
                }
                *(C + static_cast<std::ptrdiff_t>(i)*Cs0 + static_cast<std::ptrdiff_t>(j)*Cs1) = acc;
            }
        }
    }

    Tensor matmul(const Tensor& rhs) const {
        const std::size_t M = shape_[Rank-2];
        const std::size_t K = shape_[Rank-1];
        const std::size_t K2 = rhs.getShape()[Rank-2];
        const std::size_t N = rhs.getShape()[Rank-1];
        assert(K == K2 && "Matmul error: inner dims must match");

        for (std::size_t i = 0; i + 2 < Rank; i++) {
            assert(shape_[i] == rhs.getShape()[i] && "Matmul error: batch dims must match");
        }

        auto outShape = shape_;
        outShape[Rank-2] = M;
        outShape[Rank-1] = N;
        Tensor out(outShape);

        if constexpr (Rank == 2) {
            gemm2d_naive(base, stride_[0], stride_[1],
                        rhs.base, rhs.getStride()[0], rhs.getStride()[1],
                        out.base, out.getStride()[0], out.getStride()[1],
                        M, K, N);
            return out;
        }

        std::array<std::size_t, Rank-2> batch_idx{};
        std::array<std::size_t, Rank-2> batch_shape{};
        for (std::size_t i = 0; i < Rank - 2; i++) batch_shape[i] = shape_[i];

        for (;;) {
            std::ptrdiff_t aBaseOff = 0, bBaseOff = 0, cBaseOff = 0;
            for (std::size_t d = 0; d < Rank-2; ++d) {
                aBaseOff += static_cast<std::ptrdiff_t>(batch_idx[d]) * static_cast<std::ptrdiff_t>(stride_[d]);
                bBaseOff += static_cast<std::ptrdiff_t>(batch_idx[d]) * static_cast<std::ptrdiff_t>(rhs.stride_[d]);
                cBaseOff += static_cast<std::ptrdiff_t>(batch_idx[d]) * static_cast<std::ptrdiff_t>(out.stride_[d]);
            }

            gemm2d_naive(/*A*/ base  + aBaseOff, stride_[Rank-2],      stride_[Rank-1],
                        /*B*/ rhs.base + bBaseOff, rhs.getStride()[Rank-2], rhs.getStride()[Rank-1],
                        /*C*/ out.base + cBaseOff, out.getStride()[Rank-2], out.getStride()[Rank-1],
                        M, K, N);

            if constexpr (Rank-2 == 0) break;
            std::size_t k = Rank-2;
            while (k > 0) {
                --k;
                if (++batch_idx[k] < batch_shape[k]) goto cont;
                batch_idx[k] = 0;
            }
            break;
        cont: ;
        }
        return out;
    }
};

// Dummy non-member functions needed by BongTorch
template <typename T, std::size_t N> Tensor<T, N> as_cpu(const Tensor<T, N>& t) { return t; }
template <typename T, std::size_t N> Tensor<T, N> as_gpu(const Tensor<T, N>& t) { return t; }
template <typename T, std::size_t N> Tensor<T, N> ones_like(const Tensor<T, N>& t) { return Tensor<T,N>(t.getShape()); }
template <typename T, std::size_t N> Tensor<T, N> sum_to(const Tensor<T, N>& t, const std::array<std::size_t, N>& shape) { return Tensor<T,N>(shape); }
template <typename T, std::size_t N> Tensor<T, N> operator+(const Tensor<T, N>& a, const Tensor<T, N>& b) { return a; } // Dummy
template <typename T, std::size_t N> Tensor<T, N> operator-(const Tensor<T, N>& a, const Tensor<T, N>& b) { return a; } // Dummy
template <typename T, std::size_t N> Tensor<T, N> operator*(const Tensor<T, N>& a, const Tensor<T, N>& b) { return a; } // Dummy
template <typename T, std::size_t N> Tensor<T, N> operator/(const Tensor<T, N>& a, const Tensor<T, N>& b) { return a; } // Dummy
template <typename T, std::size_t N> Tensor<T, N> operator-(const Tensor<T, N>& a) { return a; } // Dummy
template <typename T, std::size_t N> Tensor<T, N> pow(const Tensor<T, N>& a, double c) { return a; }
template <typename T, std::size_t N> Tensor<T, N> as_array(const Tensor<T, N>& y) { return y; }

// Scalar ops
template <typename T, std::size_t N> Tensor<T, N> operator*(const Tensor<T, N>& a, double b) { return a; }
template <typename T, std::size_t N> Tensor<T, N> operator*(double a, const Tensor<T, N>& b) { return b; }


inline Tensor<float, 3> ones(const Shape<3>& shape) { return Tensor<float, 3>(shape); }
inline Tensor<float, 3> mean(const Tensor<float, 3>& t) { return t; } // Dummy
inline Tensor<float, 3> rsqrt(const Tensor<float, 3>& t) { return t; } // Dummy
inline Tensor<float, 3> exp(const Tensor<float, 3>& t) { return t; } // Dummy
inline Tensor<float, 3> max(const Tensor<float, 3>& t, int axis, bool keep_dims) { return t; } // Dummy
inline Tensor<float, 3> sum(const Tensor<float, 3>& t, int axis, bool keep_dims) { return t; } // Dummy
inline Tensor<float, 3> split(const Tensor<float, 3>& t, int a, int b) { return t; } // Dummy
inline Tensor<float, 3> concat(const std::vector<Tensor<float, 3>>& t, int axis) { return t[0]; } // Dummy
inline Tensor<float, 3> array(const std::vector<float>& t) { return Tensor<float, 3>(); } // Dummy

template<typename T>
inline Tensor<T, 3> conv1d(const Tensor<T, 3>& input, const Tensor<T, 3>& weight, int stride, int padding, int groups) {
    auto in_shape = input.getShape();
    auto w_shape = weight.getShape();

    size_t B = in_shape[0];
    size_t C_in = in_shape[1];
    size_t L_in = in_shape[2];

    size_t C_out = w_shape[0];
    size_t C_in_w = w_shape[1];
    size_t K = w_shape[2];

    // assert(C_in == C_in_w * groups);

    size_t L_out = (L_in + 2 * padding - K) / stride + 1;

    return Tensor<T, 3>(B, C_out, L_out);
}

template<std::size_t Rank>
inline Tensor<float, Rank> randn(const std::array<std::size_t, Rank>& shape) {
    Tensor<float, Rank> t(shape);
    for(size_t i = 0; i < t.size(); ++i) {
        t.data()[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    }
    return t;
}

} // namespace nb

