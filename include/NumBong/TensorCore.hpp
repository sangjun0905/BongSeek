#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <vector>

namespace nb {

using Shape = std::vector<std::size_t>;

template<class T, std::size_t Rank>
class Tensor {
public:
    using value_type = T;
    using shape_type = std::array<std::size_t, Rank>;
    using stride_type = std::array<std::ptrdiff_t, Rank>;
    static constexpr std::size_t rank = Rank;

private:
    std::shared_ptr<T[]> storage;
    T* base = nullptr;
    std::ptrdiff_t offset = 0;
    shape_type shape{};
    stride_type stride{};

    Tensor(std::shared_ptr<T[]> st,
           T* base_ptr,
           std::ptrdiff_t off,
           shape_type sh,
           stride_type stv)
        : storage(std::move(st)), base(base_ptr), offset(off), shape(sh), stride(stv) {}

public:
    template<class... Ds>
    explicit Tensor(Ds... dims)
    requires (sizeof...(Ds) == Rank &&
              (std::conjunction_v<std::is_integral<std::decay_t<Ds>>...>))
        : Tensor(shape_type{ static_cast<std::size_t>(dims)... }, true) {}

    static Tensor from_shape(const shape_type& shp) {
        return Tensor(shp, true);
    }

    T* data() { return base + offset; }
    const T* data() const { return base + offset; }

    const shape_type& getShape() const { return shape; }
    const stride_type& getStride() const { return stride; }

    Shape shape_vector() const { return Shape(shape.begin(), shape.end()); }

    std::size_t size() const {
        if constexpr (Rank == 0) return 1;
        std::size_t total = 1;
        for (auto d : shape) total *= d;
        return total;
    }

    std::size_t ndim() const { return Rank; }

    std::string dtype() const { return std::string(typeid(T).name()); }

    std::string shape_string() const {
        std::ostringstream oss;
        oss << '(';
        for (std::size_t i = 0; i < Rank; ++i) {
            oss << shape[i];
            if (i + 1 < Rank) oss << ", ";
        }
        oss << ')';
        return oss.str();
    }

    bool is_same_shape(const Tensor& rhs) const {
        return shape == rhs.getShape();
    }

    void fill(const T& value) {
        iterate_unary_inplace([&](T& a) { a = value; });
    }

    template<class... Ix>
    T& operator()(Ix... ix) {
        static_assert(sizeof...(Ix) == Rank, "Index count must equal rank");
        std::array<std::ptrdiff_t, Rank> idx{ static_cast<std::ptrdiff_t>(ix)... };
        std::ptrdiff_t off = offset;
        for (std::size_t i = 0; i < Rank; ++i) {
            off += idx[i] * stride[i];
        }
        return *(base + off);
    }
    template<class... Ix>
    const T& operator()(Ix... ix) const {
        return const_cast<Tensor&>(*this)(ix...);
    }

    Tensor transpose() const {
        static_assert(Rank >= 2, "transpose requires rank >= 2");
        auto sh = shape;
        auto st = stride;
        std::swap(sh[Rank - 2], sh[Rank - 1]);
        std::swap(st[Rank - 2], st[Rank - 1]);
        return Tensor(storage, base, offset, sh, st);
    }

    Tensor transpose(std::size_t a, std::size_t b) const {
        auto sh = shape;
        auto st = stride;
        std::swap(sh[a], sh[b]);
        std::swap(st[a], st[b]);
        return Tensor(storage, base, offset, sh, st);
    }

private:
    Tensor(const shape_type& shp, bool)
        : shape(shp) {
        std::size_t total = 1;
        for (auto d : shape) total *= d;
        storage = std::make_shared<T[]>(total);
        base = storage.get();
        stride[Rank - 1] = 1;
        for (std::size_t i = Rank - 1; i > 0; --i) {
            stride[i - 1] = stride[i] * static_cast<std::ptrdiff_t>(shape[i]);
        }
    }

    static void gemm2d_naive(const T* A, std::ptrdiff_t As0, std::ptrdiff_t As1,
                             const T* B, std::ptrdiff_t Bs0, std::ptrdiff_t Bs1,
                             T* C, std::ptrdiff_t Cs0, std::ptrdiff_t Cs1,
                             std::size_t M, std::size_t K, std::size_t N) {
        for (std::size_t i = 0; i < M; ++i) {
            for (std::size_t j = 0; j < N; ++j) {
                T acc{};
                for (std::size_t k = 0; k < K; ++k) {
                    const T a = *(A + static_cast<std::ptrdiff_t>(i) * As0 + static_cast<std::ptrdiff_t>(k) * As1);
                    const T b = *(B + static_cast<std::ptrdiff_t>(k) * Bs0 + static_cast<std::ptrdiff_t>(j) * Bs1);
                    acc += a * b;
                }
                *(C + static_cast<std::ptrdiff_t>(i) * Cs0 + static_cast<std::ptrdiff_t>(j) * Cs1) = acc;
            }
        }
    }

public:
    Tensor matmul(const Tensor& rhs) const {
        static_assert(Rank >= 2, "matmul requires rank >= 2");
        const std::size_t M = shape[Rank - 2];
        const std::size_t K = shape[Rank - 1];
        const std::size_t K2 = rhs.getShape()[Rank - 2];
        const std::size_t N = rhs.getShape()[Rank - 1];
        assert(K == K2 && "Matmul error: inner dims must match");

        for (std::size_t i = 0; i + 2 < Rank; ++i) {
            assert(shape[i] == rhs.getShape()[i] && "Matmul error: batch dims must match");
        }

        auto outShape = shape;
        outShape[Rank - 2] = M;
        outShape[Rank - 1] = N;
        Tensor out = Tensor::from_shape(outShape);

        if constexpr (Rank == 2) {
            gemm2d_naive(base, stride[0], stride[1],
                         rhs.base, rhs.getStride()[0], rhs.getStride()[1],
                         out.base, out.getStride()[0], out.getStride()[1],
                         M, K, N);
            return out;
        }

        std::array<std::size_t, Rank - 2> batch_idx{};
        std::array<std::size_t, Rank - 2> batch_shape{};
        for (std::size_t i = 0; i < Rank - 2; ++i) batch_shape[i] = shape[i];

        for (;;) {
            std::ptrdiff_t aBaseOff = 0, bBaseOff = 0, cBaseOff = 0;
            for (std::size_t d = 0; d < Rank - 2; ++d) {
                aBaseOff += static_cast<std::ptrdiff_t>(batch_idx[d]) * stride[d];
                bBaseOff += static_cast<std::ptrdiff_t>(batch_idx[d]) * rhs.stride[d];
                cBaseOff += static_cast<std::ptrdiff_t>(batch_idx[d]) * out.stride[d];
            }

            gemm2d_naive(base + aBaseOff, stride[Rank - 2], stride[Rank - 1],
                         rhs.base + bBaseOff, rhs.stride[Rank - 2], rhs.stride[Rank - 1],
                         out.base + cBaseOff, out.stride[Rank - 2], out.stride[Rank - 1],
                         M, K, N);

            if constexpr (Rank - 2 == 0) break;
            std::size_t k = Rank - 2;
            while (k > 0) {
                --k;
                if (++batch_idx[k] < batch_shape[k]) goto next;
                batch_idx[k] = 0;
            }
            break;
        next:;
        }
        return out;
    }

    Tensor broadcast_to(const shape_type& target) const {
        stride_type new_stride = stride;
        for (std::size_t i = 0; i < Rank; ++i) {
            if (shape[i] == target[i]) continue;
            assert(shape[i] == 1 && "broadcast_to error: incompatible dimension");
            new_stride[i] = 0;
        }
        return Tensor(storage, base, offset, target, new_stride);
    }

    Tensor sum_to(const shape_type& target) const {
        for (std::size_t i = 0; i < Rank; ++i) {
            if (target[i] == shape[i]) continue;
            assert(target[i] == 1 && "sum_to error: can only reduce to size 1 along differing axes");
        }

        Tensor out = Tensor::from_shape(target);
        out.fill(T{});

        iterate_indices(shape, [&](const auto& idx) {
            auto out_idx = idx;
            for (std::size_t axis = 0; axis < Rank; ++axis) {
                if (target[axis] == 1) out_idx[axis] = 0;
            }
            const auto self_off = offset_from_indices(idx);
            const auto out_off = out.offset_from_indices(out_idx);
            out.base[out_off] += base[self_off];
        });

        return out;
    }

    Tensor add(const Tensor& rhs) const {
        assert(is_same_shape(rhs) && "Add error: shape mismatch");
        Tensor out = Tensor::from_shape(shape);
        iterate_binary(rhs, out, [](const T& a, const T& b) { return a + b; });
        return out;
    }

    Tensor sub(const Tensor& rhs) const {
        assert(is_same_shape(rhs) && "Sub error: shape mismatch");
        Tensor out = Tensor::from_shape(shape);
        iterate_binary(rhs, out, [](const T& a, const T& b) { return a - b; });
        return out;
    }

    Tensor mul(const Tensor& rhs) const {
        assert(is_same_shape(rhs) && "Mul error: shape mismatch");
        Tensor out = Tensor::from_shape(shape);
        iterate_binary(rhs, out, [](const T& a, const T& b) { return a * b; });
        return out;
    }

    Tensor div(const Tensor& rhs) const {
        assert(is_same_shape(rhs) && "Div error: shape mismatch");
        Tensor out = Tensor::from_shape(shape);
        iterate_binary(rhs, out, [](const T& a, const T& b) { return a / b; });
        return out;
    }

    Tensor add(const T& scalar) const {
        Tensor out = Tensor::from_shape(shape);
        iterate_unary(out, [&](const T& a) { return a + scalar; });
        return out;
    }

    Tensor sub(const T& scalar) const {
        Tensor out = Tensor::from_shape(shape);
        iterate_unary(out, [&](const T& a) { return a - scalar; });
        return out;
    }

    Tensor mul(const T& scalar) const {
        Tensor out = Tensor::from_shape(shape);
        iterate_unary(out, [&](const T& a) { return a * scalar; });
        return out;
    }

    Tensor div(const T& scalar) const {
        Tensor out = Tensor::from_shape(shape);
        iterate_unary(out, [&](const T& a) { return a / scalar; });
        return out;
    }

    Tensor negate() const {
        Tensor out = Tensor::from_shape(shape);
        iterate_unary(out, [](const T& a) { return -a; });
        return out;
    }

    Tensor pow(double exponent) const {
        Tensor out = Tensor::from_shape(shape);
        iterate_unary(out, [&](const T& a) {
            return static_cast<T>(std::pow(static_cast<double>(a), exponent));
        });
        return out;
    }

    Tensor& iadd(const Tensor& rhs) {
        assert(is_same_shape(rhs) && "iadd error: shape mismatch");
        iterate_binary_inplace(rhs, [](T& a, const T& b) { a += b; });
        return *this;
    }

    Tensor& iadd(const T& scalar) {
        iterate_unary_inplace([&](T& a) { a += scalar; });
        return *this;
    }

    Tensor operator+(const Tensor& rhs) const { return add(rhs); }
    Tensor operator-(const Tensor& rhs) const { return sub(rhs); }
    Tensor operator*(const Tensor& rhs) const { return mul(rhs); }
    Tensor operator/(const Tensor& rhs) const { return div(rhs); }

    template<class U, std::enable_if_t<std::is_arithmetic_v<U>, int> = 0>
    Tensor operator+(U scalar) const { return add(static_cast<T>(scalar)); }
    template<class U, std::enable_if_t<std::is_arithmetic_v<U>, int> = 0>
    Tensor operator-(U scalar) const { return sub(static_cast<T>(scalar)); }
    template<class U, std::enable_if_t<std::is_arithmetic_v<U>, int> = 0>
    Tensor operator*(U scalar) const { return mul(static_cast<T>(scalar)); }
    template<class U, std::enable_if_t<std::is_arithmetic_v<U>, int> = 0>
    Tensor operator/(U scalar) const { return div(static_cast<T>(scalar)); }

    Tensor operator-() const { return negate(); }

    template<class U, std::enable_if_t<std::is_arithmetic_v<U>, int> = 0>
    friend Tensor operator+(U scalar, const Tensor& tensor) {
        return tensor + scalar;
    }
    template<class U, std::enable_if_t<std::is_arithmetic_v<U>, int> = 0>
    friend Tensor operator-(U scalar, const Tensor& tensor) {
        Tensor result = tensor.negate();
        result.iadd(static_cast<T>(scalar));
        return result;
    }
    template<class U, std::enable_if_t<std::is_arithmetic_v<U>, int> = 0>
    friend Tensor operator*(U scalar, const Tensor& tensor) {
        return tensor * scalar;
    }

private:
    std::ptrdiff_t offset_from_indices(const shape_type& idx) const {
        std::ptrdiff_t off = offset;
        for (std::size_t i = 0; i < Rank; ++i) {
            off += static_cast<std::ptrdiff_t>(idx[i]) * stride[i];
        }
        return off;
    }

    template<class Func>
    static void iterate_indices(const shape_type& sh, Func&& func) {
        if constexpr (Rank == 0) {
            std::array<std::size_t, 0> idx{};
            func(idx);
            return;
        }
        shape_type idx{};
        for (;;) {
            func(idx);
            std::size_t axis = Rank;
            while (axis > 0) {
                --axis;
                if (++idx[axis] < sh[axis]) goto next;
                idx[axis] = 0;
            }
            break;
        next:;
        }
    }

    template<class Func>
    void iterate_unary(Tensor& out, Func&& func) const {
        iterate_indices(shape, [&](const auto& idx) {
            const auto self_off = offset_from_indices(idx);
            const auto out_off = out.offset_from_indices(idx);
            out.base[out_off] = func(base[self_off]);
        });
    }

    template<class Func>
    void iterate_binary(const Tensor& rhs, Tensor& out, Func&& func) const {
        iterate_indices(shape, [&](const auto& idx) {
            const auto self_off = offset_from_indices(idx);
            const auto rhs_off = rhs.offset_from_indices(idx);
            const auto out_off = out.offset_from_indices(idx);
            out.base[out_off] = func(base[self_off], rhs.base[rhs_off]);
        });
    }

    template<class Func>
    void iterate_unary_inplace(Func&& func) {
        iterate_indices(shape, [&](const auto& idx) {
            const auto self_off = offset_from_indices(idx);
            func(base[self_off]);
        });
    }

    template<class Func>
    void iterate_binary_inplace(const Tensor& rhs, Func&& func) {
        iterate_indices(shape, [&](const auto& idx) {
            const auto self_off = offset_from_indices(idx);
            const auto rhs_off = rhs.offset_from_indices(idx);
            func(base[self_off], rhs.base[rhs_off]);
        });
    }
};

} // namespace nb
