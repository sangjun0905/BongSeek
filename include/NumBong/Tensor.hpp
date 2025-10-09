#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <initializer_list>
#include <istream>
#include <limits>
#include <numeric>
#include <thread>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "BFloat16.hpp"

namespace nb {

using Shape = std::vector<std::size_t>;

namespace detail {

template<typename T>
struct Accumulator {
    using type = T;
    static type convert(const T& value) { return static_cast<type>(value); }
    static T cast(const type& value) { return static_cast<T>(value); }
};

template<>
struct Accumulator<BFloat16> {
    using type = float;
    static type convert(const BFloat16& value) { return static_cast<float>(value); }
    static BFloat16 cast(const type& value) { return BFloat16(value); }
};

template<typename T>
inline double to_double(const T& value) {
    if constexpr (std::is_same_v<T, BFloat16>) {
        return static_cast<double>(static_cast<float>(value));
    } else {
        return static_cast<double>(value);
    }
}

template<std::size_t Rank>
std::size_t compute_size(const std::array<std::size_t, Rank>& shape) {
    if constexpr (Rank == 0) {
        return 1;
    } else {
        std::size_t total = 1;
        for (auto dim : shape) {
            total *= dim;
        }
        return total;
    }
}

template<std::size_t Rank>
std::array<std::size_t, Rank> compute_strides(const std::array<std::size_t, Rank>& shape) {
    std::array<std::size_t, Rank> strides{};
    if constexpr (Rank > 0) {
        strides[Rank - 1] = 1;
        for (std::size_t axis = Rank; axis-- > 0;) {
            if (axis + 1 < Rank) {
                strides[axis] = strides[axis + 1] * shape[axis + 1];
            }
        }
    }
    return strides;
}

template<std::size_t Rank>
bool has_zero_dim(const std::array<std::size_t, Rank>& shape) {
    if constexpr (Rank == 0) {
        return false;
    } else {
        for (auto dim : shape) {
            if (dim == 0) return true;
        }
        return false;
    }
}

template<std::size_t Rank, typename Func>
void for_each_index(const std::array<std::size_t, Rank>& shape, Func&& func) {
    if constexpr (Rank == 0) {
        func(std::array<std::size_t, 0>{});
        return;
    } else {
        if (has_zero_dim(shape)) return;
        std::array<std::size_t, Rank> idx{};
        while (true) {
            func(idx);
            std::size_t axis = Rank;
            while (axis > 0) {
                --axis;
                ++idx[axis];
                if (idx[axis] < shape[axis]) goto next;
                idx[axis] = 0;
            }
            break;
        next:;
        }
    }
}

template<std::size_t Rank>
std::array<std::size_t, Rank> shape_to_array(const Shape& shape) {
    if (shape.size() != Rank) {
        throw std::invalid_argument("shape_to_array: rank mismatch");
    }
    std::array<std::size_t, Rank> out{};
    for (std::size_t i = 0; i < Rank; ++i) {
        out[i] = shape[i];
    }
    return out;
}

template<std::size_t Rank>
std::array<std::size_t, Rank> shape_to_array(std::initializer_list<std::size_t> values) {
    if (values.size() != Rank) {
        throw std::invalid_argument("shape_to_array: rank mismatch");
    }
    std::array<std::size_t, Rank> out{};
    std::size_t i = 0;
    for (auto v : values) {
        out[i++] = v;
    }
    return out;
}

template<std::size_t Rank>
Shape array_to_shape(const std::array<std::size_t, Rank>& arr) {
    return Shape(arr.begin(), arr.end());
}

template<std::size_t Rank>
int normalize_axis(int axis) {
    int upper = static_cast<int>(Rank);
    if (axis < 0) axis += upper;
    if (axis < 0 || axis >= upper) {
        throw std::out_of_range("axis is out of range");
    }
    return axis;
}

template<std::size_t Rank>
std::array<std::size_t, Rank> resolve_reshape_dims(const std::vector<std::ptrdiff_t>& dims,
                                                   std::size_t total_size) {
    if (dims.size() != Rank) {
        throw std::invalid_argument("nb::reshape: dimension count must match tensor rank");
    }

    std::array<std::size_t, Rank> result{};
    std::size_t known_product = 1;
    int infer_index = -1;
    bool has_zero_dim = false;

    for (std::size_t i = 0; i < Rank; ++i) {
        const auto dim = dims[i];
        if (dim == -1) {
            if (infer_index != -1) {
                throw std::invalid_argument("nb::reshape: only one inferred dimension allowed");
            }
            infer_index = static_cast<int>(i);
            result[i] = 0; // placeholder until inference
        } else if (dim >= 0) {
            const auto as_size = static_cast<std::size_t>(dim);
            result[i] = as_size;
            if (as_size == 0) {
                has_zero_dim = true;
            } else if (known_product > 0) {
                known_product *= as_size;
            }
        } else {
            throw std::invalid_argument("nb::reshape: dimensions must be non-negative or -1");
        }
    }

    if (has_zero_dim && total_size != 0) {
        throw std::invalid_argument("nb::reshape: zero dimension requires tensor with zero elements");
    }

    if (infer_index != -1) {
        if (has_zero_dim) {
            result[infer_index] = 0;
        } else {
            if (known_product == 0 || total_size % known_product != 0) {
                throw std::invalid_argument("nb::reshape: cannot infer dimension with non-divisible size");
            }
            result[infer_index] = total_size / known_product;
        }
    } else {
        const std::size_t expected_size = has_zero_dim ? 0 : known_product;
        if (expected_size != total_size) {
            throw std::invalid_argument("nb::reshape: total size mismatch");
        }
    }

    return result;
}

} // namespace detail

template<typename T, std::size_t Rank>
class Tensor {
public:
    using value_type = T;
    using shape_type = std::array<std::size_t, Rank>;
    using stride_type = std::array<std::size_t, Rank>;

    Tensor() = default;

    template<typename... Dims>
    explicit Tensor(Dims... dims) {
        static_assert(sizeof...(Dims) == Rank, "Tensor: dimension count must match rank");
        static_assert((std::is_integral_v<std::decay_t<Dims>> && ...), "Tensor: dimensions must be integral");
        shape_type shape{ static_cast<std::size_t>(dims)... };
        init_from_shape(shape);
    }

    explicit Tensor(const shape_type& shape) {
        init_from_shape(shape);
    }

    explicit Tensor(std::initializer_list<std::size_t> dims) {
        init_from_shape(detail::shape_to_array<Rank>(dims));
    }

    Tensor(const shape_type& shape, const T& value) {
        init_from_shape(shape);
        std::fill(data_.begin(), data_.end(), value);
    }

    explicit Tensor(const Shape& shape_vec) {
        init_from_shape(detail::shape_to_array<Rank>(shape_vec));
    }

    Tensor(const Tensor&) = default;
    Tensor(Tensor&&) noexcept = default;
    Tensor& operator=(const Tensor&) = default;
    Tensor& operator=(Tensor&&) noexcept = default;

    static Tensor from_shape(const shape_type& shape) {
        return Tensor(shape);
    }

    T* data() { return data_.empty() ? nullptr : data_.data(); }
    const T* data() const { return data_.empty() ? nullptr : data_.data(); }

    const shape_type& getShape() const { return shape_; }
    const stride_type& getStride() const { return stride_; }

    Shape shape_vector() const { return detail::array_to_shape(shape_); }
    Shape shape() const { return detail::array_to_shape(shape_); }

    std::size_t size() const { return data_.size(); }
    std::size_t ndim() const { return Rank; }

    std::string dtype() const { return std::string(typeid(T).name()); }

    std::string shape_string() const {
        std::ostringstream oss;
        oss << '(';
        for (std::size_t i = 0; i < Rank; ++i) {
            oss << shape_[i];
            if (i + 1 < Rank) oss << ", ";
        }
        oss << ')';
        return oss.str();
    }

    bool is_same_shape(const Tensor& other) const { return shape_ == other.shape_; }

    void fill(const T& value) {
        std::fill(data_.begin(), data_.end(), value);
    }

    template<typename... Ix>
    T& operator()(Ix... indices) {
        static_assert(sizeof...(Ix) == Rank, "Tensor: index count must match rank");
        shape_type idx{ static_cast<std::size_t>(indices)... };
        return data_.at(offset(idx));
    }

    template<typename... Ix>
    const T& operator()(Ix... indices) const {
        static_assert(sizeof...(Ix) == Rank, "Tensor: index count must match rank");
        shape_type idx{ static_cast<std::size_t>(indices)... };
        return data_.at(offset(idx));
    }

    T& operator[](const shape_type& idx) {
        return data_.at(offset(idx));
    }

    const T& operator[](const shape_type& idx) const {
        return data_.at(offset(idx));
    }

    T& operator[](const Shape& idx) {
        return (*this)[detail::shape_to_array<Rank>(idx)];
    }

    const T& operator[](const Shape& idx) const {
        return (*this)[detail::shape_to_array<Rank>(idx)];
    }

    T& operator[](std::initializer_list<std::size_t> idx) {
        return (*this)[detail::shape_to_array<Rank>(idx)];
    }

    const T& operator[](std::initializer_list<std::size_t> idx) const {
        return (*this)[detail::shape_to_array<Rank>(idx)];
    }

    Tensor transpose() const {
        static_assert(Rank >= 2, "transpose requires rank >= 2");
        return transpose(Rank - 2, Rank - 1);
    }

    Tensor transpose(std::size_t axis_a, std::size_t axis_b) const {
        if (axis_a >= Rank || axis_b >= Rank) {
            throw std::out_of_range("Tensor::transpose axis out of range");
        }
        if (axis_a == axis_b) return *this;

        shape_type new_shape = shape_;
        std::swap(new_shape[axis_a], new_shape[axis_b]);

        Tensor result(new_shape);
        detail::for_each_index<Rank>(shape_, [&](const auto& idx) {
            auto out_idx = idx;
            std::swap(out_idx[axis_a], out_idx[axis_b]);
            result[out_idx] = (*this)[idx];
        });
        return result;
    }

    Tensor broadcast_to(const shape_type& target) const {
        for (std::size_t axis = 0; axis < Rank; ++axis) {
            if (shape_[axis] == target[axis] || shape_[axis] == 1) continue;
            throw std::invalid_argument("Tensor::broadcast_to: incompatible dimensions");
        }

        Tensor result(target);
        detail::for_each_index<Rank>(target, [&](const auto& idx) {
            shape_type src_idx = idx;
            for (std::size_t axis = 0; axis < Rank; ++axis) {
                if (shape_[axis] == 1) src_idx[axis] = 0;
            }
            result[idx] = (*this)[src_idx];
        });
        return result;
    }

    Tensor sum_to(const shape_type& target) const {
        for (std::size_t axis = 0; axis < Rank; ++axis) {
            if (target[axis] == shape_[axis] || target[axis] == 1) continue;
            throw std::invalid_argument("Tensor::sum_to: incompatible target shape");
        }

        Tensor result(target);
        result.fill(T{});
        detail::for_each_index<Rank>(shape_, [&](const auto& idx) {
            auto out_idx = idx;
            for (std::size_t axis = 0; axis < Rank; ++axis) {
                if (target[axis] == 1) out_idx[axis] = 0;
            }
            result[out_idx] += (*this)[idx];
        });
        return result;
    }

    Tensor add(const Tensor& rhs) const {
        require_same_shape(rhs);
        Tensor out(shape_);
        const std::size_t total = size();
        const T* lhs_ptr = data();
        const T* rhs_ptr = rhs.data();
        T* dst = out.data();
        for (std::size_t i = 0; i < total; ++i) {
            dst[i] = lhs_ptr[i] + rhs_ptr[i];
        }
        return out;
    }

    Tensor sub(const Tensor& rhs) const {
        require_same_shape(rhs);
        Tensor out(shape_);
        const std::size_t total = size();
        const T* lhs_ptr = data();
        const T* rhs_ptr = rhs.data();
        T* dst = out.data();
        for (std::size_t i = 0; i < total; ++i) {
            dst[i] = lhs_ptr[i] - rhs_ptr[i];
        }
        return out;
    }

    Tensor mul(const Tensor& rhs) const {
        require_same_shape(rhs);
        Tensor out(shape_);
        const std::size_t total = size();
        const T* lhs_ptr = data();
        const T* rhs_ptr = rhs.data();
        T* dst = out.data();
        for (std::size_t i = 0; i < total; ++i) {
            dst[i] = lhs_ptr[i] * rhs_ptr[i];
        }
        return out;
    }

    Tensor div(const Tensor& rhs) const {
        require_same_shape(rhs);
        Tensor out(shape_);
        const std::size_t total = size();
        const T* lhs_ptr = data();
        const T* rhs_ptr = rhs.data();
        T* dst = out.data();
        for (std::size_t i = 0; i < total; ++i) {
            dst[i] = lhs_ptr[i] / rhs_ptr[i];
        }
        return out;
    }

    Tensor add(const T& scalar) const {
        Tensor out(shape_);
        const std::size_t total = size();
        const T* src = data();
        T* dst = out.data();
        for (std::size_t i = 0; i < total; ++i) {
            dst[i] = src[i] + scalar;
        }
        return out;
    }

    Tensor sub(const T& scalar) const {
        Tensor out(shape_);
        const std::size_t total = size();
        const T* src = data();
        T* dst = out.data();
        for (std::size_t i = 0; i < total; ++i) {
            dst[i] = src[i] - scalar;
        }
        return out;
    }

    Tensor mul(const T& scalar) const {
        Tensor out(shape_);
        const std::size_t total = size();
        const T* src = data();
        T* dst = out.data();
        for (std::size_t i = 0; i < total; ++i) {
            dst[i] = src[i] * scalar;
        }
        return out;
    }

    Tensor div(const T& scalar) const {
        Tensor out(shape_);
        const std::size_t total = size();
        const T* src = data();
        T* dst = out.data();
        const T inv = static_cast<T>(1) / scalar;
        for (std::size_t i = 0; i < total; ++i) {
            dst[i] = src[i] * inv;
        }
        return out;
    }

    Tensor neg() const {
        Tensor out(shape_);
        const std::size_t total = size();
        const T* src = data();
        T* dst = out.data();
        for (std::size_t i = 0; i < total; ++i) {
            dst[i] = -src[i];
        }
        return out;
    }

    Tensor pow(double exponent) const {
        Tensor out(shape_);
        const std::size_t total = size();
        const T* src = data();
        T* dst = out.data();
        for (std::size_t i = 0; i < total; ++i) {
            dst[i] = static_cast<T>(std::pow(static_cast<double>(src[i]), exponent));
        }
        return out;
    }

    Tensor& iadd(const Tensor& rhs) {
        require_same_shape(rhs);
        const std::size_t total = size();
        T* dst = data();
        const T* src = rhs.data();
        for (std::size_t i = 0; i < total; ++i) {
            dst[i] += src[i];
        }
        return *this;
    }

    Tensor& iadd(const T& scalar) {
        T* dst = data();
        const std::size_t total = size();
        for (std::size_t i = 0; i < total; ++i) {
            dst[i] += scalar;
        }
        return *this;
    }

    void loadWeight(std::istream& in, std::streamoff start_offset, std::streamoff end_offset) {
        static_assert(std::is_trivially_copyable_v<T>,
                      "Tensor::loadWeight requires a trivially copyable value type");

        if (start_offset < 0 || end_offset < 0) {
            throw std::invalid_argument("Tensor::loadWeight: offsets must be non-negative");
        }
        if (end_offset < start_offset) {
            throw std::invalid_argument("Tensor::loadWeight: end_offset must be >= start_offset");
        }

        const std::size_t element_count = size();
        const std::size_t byte_count = element_count * sizeof(T);
        const std::streamoff span = end_offset - start_offset;

        if (span != static_cast<std::streamoff>(byte_count)) {
            throw std::invalid_argument("Tensor::loadWeight: byte range and tensor size mismatch");
        }
        if (byte_count == 0) {
            return;
        }
        if (static_cast<std::uint64_t>(byte_count) >
            static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max())) {
            throw std::overflow_error("Tensor::loadWeight: tensor byte size exceeds streamsize max");
        }
        if (!in) {
            throw std::runtime_error("Tensor::loadWeight: input stream is not ready");
        }

        in.seekg(start_offset, std::ios::beg);
        if (!in) {
            throw std::runtime_error("Tensor::loadWeight: failed to seek to start_offset");
        }

        const auto expected = static_cast<std::streamsize>(byte_count);
        in.read(reinterpret_cast<char*>(data_.data()), expected);
        if (in.gcount() != expected || !in) {
            throw std::runtime_error("Tensor::loadWeight: failed to read expected number of bytes");
        }
    }

    Tensor matmul(const Tensor& rhs) const {
        static_assert(Rank >= 2, "Tensor::matmul requires rank >= 2");

        const auto& lhs_shape = shape_;
        const auto& rhs_shape = rhs.shape_;

        const std::size_t M = lhs_shape[Rank - 2];
        const std::size_t K = lhs_shape[Rank - 1];
        const std::size_t K_rhs = rhs_shape[Rank - 2];
        const std::size_t N = rhs_shape[Rank - 1];

        if (K != K_rhs) {
            throw std::invalid_argument("Tensor::matmul: inner dimensions must match");
        }

        for (std::size_t axis = 0; axis + 2 < Rank; ++axis) {
            if (lhs_shape[axis] != rhs_shape[axis]) {
                throw std::invalid_argument("Tensor::matmul: batch dimensions must align");
            }
        }

        shape_type out_shape = lhs_shape;
        out_shape[Rank - 1] = N;
        Tensor out(out_shape);

        if (M == 0 || N == 0 || K == 0 || detail::has_zero_dim(out_shape)) {
            return out;
        }

        const auto lhs_stride = stride_;
        const auto rhs_stride = rhs.getStride();
        const auto out_stride = out.getStride();

        const std::size_t As0 = lhs_stride[Rank - 2];
        const std::size_t As1 = lhs_stride[Rank - 1];
        const std::size_t Bs0 = rhs_stride[Rank - 2];
        const std::size_t Bs1 = rhs_stride[Rank - 1];
        const std::size_t Cs0 = out_stride[Rank - 2];
        const std::size_t Cs1 = out_stride[Rank - 1];

        auto compute_range = [&](const T* A_ptr,
                                 const T* B_ptr,
                                 T* C_ptr,
                                 std::size_t row_begin,
                                 std::size_t row_end) {
            using Acc = typename detail::Accumulator<T>::type;
            for (std::size_t i = row_begin; i < row_end; ++i) {
                const T* a_row = A_ptr + i * As0;
                T* c_row = C_ptr + i * Cs0;
                for (std::size_t j = 0; j < N; ++j) {
                    Acc sum = Acc{};
                    const T* b_col = B_ptr + j * Bs1;
                    for (std::size_t k = 0; k < K; ++k) {
                        const Acc a = detail::Accumulator<T>::convert(*(a_row + k * As1));
                        const Acc b = detail::Accumulator<T>::convert(*(b_col + k * Bs0));
                        sum += a * b;
                    }
                    *(c_row + j * Cs1) = detail::Accumulator<T>::cast(sum);
                }
            }
        };

        auto run_matmul = [&](const T* A_ptr, const T* B_ptr, T* C_ptr) {
            const std::size_t work_estimate = M * N * K;
            std::size_t thread_count = std::thread::hardware_concurrency();
            if (thread_count == 0) thread_count = 1;
            thread_count = std::min<std::size_t>(thread_count, M);
            const bool use_parallel = thread_count > 1 && work_estimate >= 8192;

            if (!use_parallel) {
                compute_range(A_ptr, B_ptr, C_ptr, 0, M);
                return;
            }

            const std::size_t rows_per_thread = (M + thread_count - 1) / thread_count;
            std::vector<std::thread> workers;
            workers.reserve(thread_count - 1);

            std::size_t row_begin = 0;
            for (std::size_t t = 0; t + 1 < thread_count && row_begin < M; ++t) {
                const std::size_t row_end = std::min(row_begin + rows_per_thread, M);
                workers.emplace_back(compute_range, A_ptr, B_ptr, C_ptr, row_begin, row_end);
                row_begin = row_end;
            }

            if (row_begin < M) {
                compute_range(A_ptr, B_ptr, C_ptr, row_begin, M);
            }

            for (auto& worker : workers) {
                worker.join();
            }
        };

        if constexpr (Rank == 2) {
            run_matmul(data(), rhs.data(), out.data());
        } else {
            std::array<std::size_t, Rank - 2> batch_shape{};
            for (std::size_t axis = 0; axis < Rank - 2; ++axis) {
                batch_shape[axis] = lhs_shape[axis];
            }

            detail::for_each_index<Rank - 2>(batch_shape, [&](const auto& batch_idx) {
                std::size_t a_base = 0;
                std::size_t b_base = 0;
                std::size_t c_base = 0;
                for (std::size_t axis = 0; axis < Rank - 2; ++axis) {
                    a_base += batch_idx[axis] * lhs_stride[axis];
                    b_base += batch_idx[axis] * rhs_stride[axis];
                    c_base += batch_idx[axis] * out_stride[axis];
                }

                run_matmul(data() + a_base, rhs.data() + b_base, out.data() + c_base);
            });
        }
        return out;
    }

    Tensor operator+(const Tensor& rhs) const { return add(rhs); }
    Tensor operator-(const Tensor& rhs) const { return sub(rhs); }
    Tensor operator*(const Tensor& rhs) const { return mul(rhs); }
    Tensor operator/(const Tensor& rhs) const { return div(rhs); }

    Tensor operator+(const T& scalar) const { return add(scalar); }
    Tensor operator-(const T& scalar) const { return sub(scalar); }
    Tensor operator*(const T& scalar) const { return mul(scalar); }
    Tensor operator/(const T& scalar) const { return div(scalar); }

    Tensor operator-() const { return neg(); }
    Tensor operator^(double exponent) const { return pow(exponent); }

    template<typename U, std::enable_if_t<std::is_arithmetic_v<U>, int> = 0>
    friend Tensor operator+(U scalar, const Tensor& tensor) {
        return tensor.add(static_cast<T>(scalar));
    }

    template<typename U, std::enable_if_t<std::is_arithmetic_v<U>, int> = 0>
    friend Tensor operator-(U scalar, const Tensor& tensor) {
        Tensor result = tensor.neg();
        result.iadd(static_cast<T>(scalar));
        return result;
    }

    template<typename U, std::enable_if_t<std::is_arithmetic_v<U>, int> = 0>
    friend Tensor operator*(U scalar, const Tensor& tensor) {
        return tensor.mul(static_cast<T>(scalar));
    }

    template<typename U, std::enable_if_t<std::is_arithmetic_v<U>, int> = 0>
    friend Tensor operator/(U scalar, const Tensor& tensor) {
        Tensor result(tensor.shape_);
        const std::size_t total = tensor.size();
        const T* src = tensor.data();
        T* dst = result.data();
        const double numerator = static_cast<double>(scalar);
        for (std::size_t i = 0; i < total; ++i) {
            dst[i] = static_cast<T>(numerator / static_cast<double>(src[i]));
        }
        return result;
    }

private:
    shape_type shape_{};
    stride_type stride_{};
    std::vector<T> data_{};

    void init_from_shape(const shape_type& shape) {
        shape_ = shape;
        stride_ = detail::compute_strides(shape_);
        const std::size_t total = detail::compute_size(shape_);
        data_.assign(total, T{});
    }

    std::size_t offset(const shape_type& idx) const {
        std::size_t off = 0;
        for (std::size_t axis = 0; axis < Rank; ++axis) {
            if (idx[axis] >= shape_[axis]) {
                throw std::out_of_range("Tensor: index out of range");
            }
            off += idx[axis] * stride_[axis];
        }
        return off;
    }

    void require_same_shape(const Tensor& rhs) const {
        if (!is_same_shape(rhs)) {
            throw std::invalid_argument("Tensor: shape mismatch");
        }
    }
};

template<typename T, std::size_t Rank>
Tensor<T, Rank> broadcast_to(const Tensor<T, Rank>& tensor, const Shape& target) {
    return tensor.broadcast_to(detail::shape_to_array<Rank>(target));
}

template<typename T, std::size_t Rank>
Tensor<T, Rank> broadcast_to(const Tensor<T, Rank>& tensor, const typename Tensor<T, Rank>::shape_type& target) {
    return tensor.broadcast_to(target);
}

template<typename T, std::size_t Rank>
Tensor<T, Rank> sum_to(const Tensor<T, Rank>& tensor, const Shape& target) {
    return tensor.sum_to(detail::shape_to_array<Rank>(target));
}

template<typename T, std::size_t Rank>
Tensor<T, Rank> sum_to(const Tensor<T, Rank>& tensor, const typename Tensor<T, Rank>::shape_type& target) {
    return tensor.sum_to(target);
}

template<typename T, std::size_t Rank>
Tensor<T, Rank> ones_like(const Tensor<T, Rank>& tensor) {
    Tensor<T, Rank> out(tensor.getShape());
    out.fill(static_cast<T>(1));
    return out;
}

template<typename T, std::size_t Rank>
Tensor<T, Rank> zeros_like(const Tensor<T, Rank>& tensor) {
    return Tensor<T, Rank>(tensor.getShape());
}

template<typename T, std::size_t Rank>
Tensor<T, Rank> exp(const Tensor<T, Rank>& tensor) {
    Tensor<T, Rank> out(tensor.getShape());
    const std::size_t total = tensor.size();
    const T* src = tensor.data();
    T* dst = out.data();
    for (std::size_t i = 0; i < total; ++i) {
        dst[i] = static_cast<T>(std::exp(static_cast<double>(src[i])));
    }
    return out;
}

template<typename T, std::size_t Rank>
Tensor<T, Rank> rsqrt(const Tensor<T, Rank>& tensor) {
    Tensor<T, Rank> out(tensor.getShape());
    const std::size_t total = tensor.size();
    const T* src = tensor.data();
    T* dst = out.data();
    for (std::size_t i = 0; i < total; ++i) {
        dst[i] = static_cast<T>(1.0 / std::sqrt(static_cast<double>(src[i])));
    }
    return out;
}

template<typename T, std::size_t Rank>
Tensor<T, Rank> pow(const Tensor<T, Rank>& tensor, double exponent) {
    return tensor.pow(exponent);
}

template<typename TensorType>
TensorType as_array(const TensorType& tensor) {
    return tensor;
}

template<typename TensorType>
TensorType as_cpu(const TensorType& tensor) {
    return tensor;
}

template<typename TensorType>
TensorType as_gpu(const TensorType& tensor) {
    return tensor;
}

template<typename T, std::size_t Rank>
Tensor<T, Rank> reshape(const Tensor<T, Rank>& tensor,
                        const typename Tensor<T, Rank>::shape_type& target_shape) {
    const std::size_t new_size = detail::compute_size(target_shape);
    if (new_size != tensor.size()) {
        throw std::invalid_argument("nb::reshape: total size mismatch");
    }

    Tensor<T, Rank> out(target_shape);
    if (new_size > 0) {
        std::copy(tensor.data(), tensor.data() + new_size, out.data());
    }
    return out;
}

template<typename T, std::size_t Rank>
Tensor<T, Rank> reshape(const Tensor<T, Rank>& tensor, const Shape& dims) {
    return reshape(tensor, detail::shape_to_array<Rank>(dims));
}

template<typename T, std::size_t Rank>
Tensor<T, Rank> reshape(const Tensor<T, Rank>& tensor, const std::vector<std::ptrdiff_t>& dims) {
    return reshape(tensor, detail::resolve_reshape_dims<Rank>(dims, tensor.size()));
}

template<typename T, std::size_t Rank>
Tensor<T, Rank> reshape(const Tensor<T, Rank>& tensor, std::initializer_list<std::ptrdiff_t> dims) {
    return reshape(tensor, std::vector<std::ptrdiff_t>(dims));
}

template<typename T, std::size_t Rank>
Tensor<T, Rank> repeat(const Tensor<T, Rank>& tensor, std::size_t repeats, int axis) {
    if (repeats == 0) {
        auto out_shape = tensor.getShape();
        out_shape[static_cast<std::size_t>(detail::normalize_axis<Rank>(axis))] = 0;
        return Tensor<T, Rank>(out_shape);
    }

    const int normalized_axis = detail::normalize_axis<Rank>(axis);
    const std::size_t axis_index = static_cast<std::size_t>(normalized_axis);

    auto input_shape = tensor.getShape();
    const std::size_t axis_dim = input_shape[axis_index];
    auto out_shape = input_shape;
    out_shape[axis_index] = axis_dim * repeats;

    Tensor<T, Rank> out(out_shape);
    if (axis_dim == 0) {
        return out;
    }

    detail::for_each_index<Rank>(input_shape, [&](const auto& idx) {
        for (std::size_t r = 0; r < repeats; ++r) {
            auto out_idx = idx;
            out_idx[axis_index] = idx[axis_index] + r * axis_dim;
            out[out_idx] = tensor[idx];
        }
    });

    return out;
}

template<typename T, std::size_t Rank>
Tensor<T, Rank> sum(const Tensor<T, Rank>& tensor, int axis, bool keepdims) {
    const int ax = detail::normalize_axis<Rank>(axis);
    if (!keepdims) {
        throw std::invalid_argument("nb::sum: keepdims=false is not supported for fixed-rank tensors");
    }

    auto out_shape = tensor.getShape();
    out_shape[ax] = 1;

    Tensor<T, Rank> out(out_shape);
    out.fill(T{});

    detail::for_each_index<Rank>(tensor.getShape(), [&](const auto& idx) {
        auto out_idx = idx;
        out_idx[ax] = 0;
        out[out_idx] += tensor[idx];
    });

    return out;
}

template<typename T, std::size_t Rank>
Tensor<T, Rank> max(const Tensor<T, Rank>& tensor, int axis, bool keepdims) {
    const int ax = detail::normalize_axis<Rank>(axis);
    if (!keepdims) {
        throw std::invalid_argument("nb::max: keepdims=false is not supported for fixed-rank tensors");
    }

    auto out_shape = tensor.getShape();
    out_shape[ax] = 1;

    Tensor<T, Rank> out(out_shape);

    const std::size_t axis_dim = tensor.getShape()[ax];
    detail::for_each_index<Rank>(out_shape, [&](const auto& base_idx) {
        auto scan_idx = base_idx;
        bool has_value = false;
        T best{};
        for (std::size_t i = 0; i < axis_dim; ++i) {
            scan_idx[ax] = i;
            const T& value = tensor[scan_idx];
            if (!has_value || detail::to_double(value) > detail::to_double(best)) {
                best = value;
                has_value = true;
            }
        }
        out[base_idx] = has_value ? best : T{};
    });

    return out;
}

template<typename T, std::size_t Rank>
Tensor<T, Rank> mean(const Tensor<T, Rank>& tensor, int axis = -1, bool keepdims = true) {
    const int ax = detail::normalize_axis<Rank>(axis);
    auto total = sum(tensor, ax, true);
    const auto count = tensor.getShape()[ax];
    if (count == 0) return total;
    return total / static_cast<T>(count);
}

template<typename T, std::size_t Rank>
Tensor<T, Rank> split(const Tensor<T, Rank>& tensor, int index, std::size_t chunk) {
    if (index < 0) {
        throw std::out_of_range("nb::split: index must be non-negative");
    }
    constexpr std::size_t axis = Rank - 1;
    const auto& shape = tensor.getShape();
    const std::size_t start = static_cast<std::size_t>(index) * chunk;
    if (start + chunk > shape[axis]) {
        throw std::out_of_range("nb::split: slice exceeds dimension");
    }

    auto out_shape = shape;
    out_shape[axis] = chunk;
    Tensor<T, Rank> out(out_shape);

    detail::for_each_index<Rank>(out_shape, [&](const auto& idx) {
        auto src_idx = idx;
        src_idx[axis] = idx[axis] + start;
        out[idx] = tensor[src_idx];
    });

    return out;
}

template<typename T, std::size_t Rank>
Tensor<T, Rank> concat(const std::vector<Tensor<T, Rank>>& tensors, int axis) {
    if (tensors.empty()) {
        throw std::invalid_argument("nb::concat: tensors list must not be empty");
    }

    const int ax = detail::normalize_axis<Rank>(axis);
    auto out_shape = tensors.front().getShape();
    out_shape[ax] = 0;

    for (const auto& tensor : tensors) {
        const auto& shape = tensor.getShape();
        for (std::size_t i = 0; i < Rank; ++i) {
            if (i == static_cast<std::size_t>(ax)) continue;
            if (shape[i] != out_shape[i] && out_shape[i] != 0) {
                throw std::invalid_argument("nb::concat: shapes must match except along the concatenation axis");
            }
        }
        out_shape[ax] += shape[ax];
    }

    Tensor<T, Rank> out(out_shape);
    std::size_t offset = 0;

    for (const auto& tensor : tensors) {
        detail::for_each_index<Rank>(tensor.getShape(), [&](const auto& idx) {
            auto out_idx = idx;
            out_idx[ax] = idx[ax] + offset;
            out[out_idx] = tensor[idx];
        });
        offset += tensor.getShape()[ax];
    }

    return out;
}

} // namespace nb
