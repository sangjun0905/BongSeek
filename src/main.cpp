#include <iostream>
#include <vector>
#include <array>
#include <cassert>
#include <type_traits>

template<class T, std::size_t Rank>
class Tensor {
private:
    std::shared_ptr<T[]> storage;
    T* base = nullptr;
    std::ptrdiff_t offset = 0;
    std::array<std::size_t, Rank> shape{};
    std::array<std::ptrdiff_t, Rank> stride{};

    Tensor(
        std::shared_ptr<T[]> st, 
        T* base, 
        std::ptrdiff_t off, 
        std::array<std::size_t, Rank> sh, 
        std::array<std::ptrdiff_t, Rank> stv)
        : storage(std::move(st)), base(base), offset(off), shape(sh), stride(stv) {}
    
    Tensor(const std::array<std::size_t, Rank>& shp) : shape(shp) {
        std::size_t total = 1; for (auto d : shape) total *= d;
        storage = std::make_shared<T[]>(total);
        base = storage.get();
        stride[Rank-1] = 1;
        for (std::size_t i = Rank - 1; i > 0; i--) {
            stride[i - 1] = stride[i] * shape[i];
        }
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
public:
    template<class... Ds>
    explicit Tensor(Ds... dims)
    requires (sizeof...(Ds) == Rank && (std::conjunction_v<std::is_integral<std::decay_t<Ds>>...>))
    : shape{ static_cast<std::size_t>(dims)... } {
        storage = std::make_shared<T[]>((std::size_t{1} * ... * static_cast<std::size_t>(dims)));
        base = storage.get();
        stride[Rank - 1] = 1; // Last element is 1.
        for (std::size_t i = Rank - 1; i > 0; i--) {
            stride[i - 1] = stride[i] * static_cast<std::ptrdiff_t>(shape[i]);
        }
    }

    T* data() { return base + offset; }
    const T* data() const { return base + offset; }
    
    const auto& getShape() const { return shape; }
    const auto& getStride() const { return stride; }
        
    template<class... Ix>
    T& operator()(Ix... ix) {
        std::array<std::ptrdiff_t, Rank> idx{ static_cast<std::ptrdiff_t>(ix)... };
        std::ptrdiff_t off = offset;
        for (std::size_t i = 0; i < Rank; i++) {
            off += idx[i] * stride[i];
        }
        return *(base + off); 
    }
    template<class... Ix>
    const T& operator()(Ix... ix) const { 
        return const_cast<Tensor&>(*this)(ix...); 
    }

    Tensor transpose() const {
        auto sh = shape;
        auto st = stride;
        std::swap(sh[Rank-2], sh[Rank-1]);
        std::swap(st[Rank-2], st[Rank-1]);
        return Tensor(storage, base, 0, sh, st);
    }
    Tensor transpose(std::size_t a, std::size_t b) const {
        auto sh = shape;
        auto st = stride;
        std::swap(sh[a], sh[b]);
        std::swap(st[a], st[b]);
        return Tensor(storage, base, 0, sh, st);
    }

    Tensor matmul(const Tensor& rhs) const {
        const std::size_t M = shape[Rank-2];
        const std::size_t K = shape[Rank-1];
        const std::size_t K2 = rhs.getShape()[Rank-2];
        const std::size_t N = rhs.getShape()[Rank-1];
        assert(K == K2 && "Matmul error: inner dims must match");

        for (std::size_t i = 0; i + 2 < Rank; i++) {
            assert(shape[i] == rhs.getShape()[i] && "Matmul error: batch dims must match");
        }

        auto outShape = shape;
        outShape[Rank-2] = M;
        outShape[Rank-1] = N;
        Tensor out(outShape);

        if constexpr (Rank == 2) {
            gemm2d_naive(base, stride[0], stride[1],
                        rhs.base, rhs.getStride()[0], rhs.getStride()[1],
                        out.base, out.getStride()[0], out.getStride()[1],
                        M, K, N);
            return out;
        }

        std::array<std::size_t, Rank-2> batch_idx{};
        std::array<std::size_t, Rank-2> batch_shape{};
        for (std::size_t i = 0; i < Rank - 2; i++) batch_shape[i] = shape[i];

        for (;;) {
            std::ptrdiff_t aBaseOff = 0, bBaseOff = 0, cBaseOff = 0;
            for (std::size_t d = 0; d < Rank-2; ++d) {
                aBaseOff += static_cast<std::ptrdiff_t>(batch_idx[d]) * static_cast<std::ptrdiff_t>(stride[d]);
                bBaseOff += static_cast<std::ptrdiff_t>(batch_idx[d]) * static_cast<std::ptrdiff_t>(rhs.stride[d]);
                cBaseOff += static_cast<std::ptrdiff_t>(batch_idx[d]) * static_cast<std::ptrdiff_t>(out.stride[d]);
            }

            gemm2d_naive(/*A*/ base  + aBaseOff, stride[Rank-2],      stride[Rank-1],
                        /*B*/ rhs.base + bBaseOff, rhs.getStride()[Rank-2], rhs.getStride()[Rank-1],
                        /*C*/ out.base + cBaseOff, out.getStride()[Rank-2], out.getStride()[Rank-1],
                        M, K, N);

            // ++batch
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

int main() {
    Tensor<double, 3> m1(1000, 100, 100);
    Tensor<double, 3> m2(1000, 100, 100);

    m1(40, 2, 1) = 50.0;
    m2(40, 1, 2) = 40.0;
    m1 = m1.matmul(m2);

    std::cout << m1(40, 2, 2) << std::endl;

    return 0;
}