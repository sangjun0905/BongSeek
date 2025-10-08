// #include "bongpy.hpp" ����: nb::Array, nb::ones_like, nb::sum_to, nb::reshape, nb::transpose, nb::pow, nb::as_array
#include "Core.hpp"
#include "RMSNormFunction.hpp"

namespace bs {

    class RMSNorm : public Module {
public:
    std::shared_ptr<Parameter> weight;

    RMSNorm(int dim) {
        weight = Parameter::create(nb::ones({ dim }));
    }

    std::shared_ptr<Variable> forward(const std::shared_ptr<Variable>& x) {
        return rms_norm(x, weight);
    }
};
}