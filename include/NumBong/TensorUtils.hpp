#pragma once

#include "TensorCore.hpp"

namespace nb {

template<std::size_t Rank>
std::array<std::size_t, Rank> shape_to_array(const Shape& shape) {
    assert(shape.size() == Rank && "shape_to_array error: rank mismatch");
    std::array<std::size_t, Rank> out{};
    std::copy(shape.begin(), shape.end(), out.begin());
    return out;
}

template<std::size_t Rank>
Shape array_to_shape(const std::array<std::size_t, Rank>& arr) {
    return Shape(arr.begin(), arr.end());
}

template<class T, std::size_t Rank>
Tensor<T, Rank> broadcast_to(const Tensor<T, Rank>& tensor, const typename Tensor<T, Rank>::shape_type& target_shape) {
    return tensor.broadcast_to(target_shape);
}

template<class T, std::size_t Rank>
Tensor<T, Rank> broadcast_to(const Tensor<T, Rank>& tensor, const Shape& target_shape) {
    return tensor.broadcast_to(shape_to_array<Rank>(target_shape));
}

template<class T, std::size_t Rank>
Tensor<T, Rank> sum_to(const Tensor<T, Rank>& tensor, const typename Tensor<T, Rank>::shape_type& target_shape) {
    return tensor.sum_to(target_shape);
}

template<class T, std::size_t Rank>
Tensor<T, Rank> sum_to(const Tensor<T, Rank>& tensor, const Shape& target_shape) {
    return tensor.sum_to(shape_to_array<Rank>(target_shape));
}

template<class T, std::size_t Rank>
Tensor<T, Rank> ones_like(const Tensor<T, Rank>& tensor) {
    auto out = Tensor<T, Rank>::from_shape(tensor.getShape());
    out.fill(static_cast<T>(1));
    return out;
}

template<class T, std::size_t Rank>
Tensor<T, Rank> zeros_like(const Tensor<T, Rank>& tensor) {
    auto out = Tensor<T, Rank>::from_shape(tensor.getShape());
    out.fill(T{});
    return out;
}

template<class T, std::size_t Rank>
Tensor<T, Rank> pow(const Tensor<T, Rank>& tensor, double exponent) {
    return tensor.pow(exponent);
}

template<class TensorType>
TensorType as_array(const TensorType& tensor) {
    return tensor;
}

template<class TensorType>
TensorType as_cpu(const TensorType& tensor) {
    return tensor;
}

template<class TensorType>
TensorType as_gpu(const TensorType& tensor) {
    return tensor;
}

} // namespace nb

