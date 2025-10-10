#pragma once

#include "Core.hpp"

namespace bs {

class RMSNorm : public Module {
private:
    std::shared_ptr<Parameter> weight;
    nb::BFloat16 epsilon_{nb::BFloat16(1e-5f)};
    int dim_;

public:
    RMSNorm() {};

    RMSNorm(int dim) : dim_(dim) {
        TensorShape weight_shape = {1, 1, static_cast<size_t>(dim)};
        auto weight_tensor = Tensor(weight_shape);
        weight_tensor.fill(TensorValueType(1.0f)); // Initialize with ones
        weight = Parameter::create(weight_tensor, "weight");
        register_parameter("weight", weight);
    }

    std::shared_ptr<Variable> forward(const std::shared_ptr<Variable>& x_var) override {
        const auto& x = x_var->data;
        auto x_shape = x.getShape();
        auto B = x_shape[0];
        auto S = x_shape[1];
        auto D = x_shape[2];

        if (D != static_cast<size_t>(dim_)) {
            throw std::runtime_error("RMSNorm: input dimension mismatch.");
        }

        Tensor output(x_shape);

        const nb::BFloat16 dim_b(dim_);
        const nb::BFloat16 one(1.0f);

        for (size_t b = 0; b < B; ++b) {
            for (size_t s = 0; s < S; ++s) {
                nb::BFloat16 sum_sq(0.0f);
                for (size_t d = 0; d < D; ++d) {
                    const nb::BFloat16 val = x(b, s, d);
                    sum_sq += val * val;
                }
                const nb::BFloat16 mean_sq = sum_sq / dim_b;
                const nb::BFloat16 denom = nb::bfloat16_sqrt(mean_sq + epsilon_);
                const nb::BFloat16 inv_rms = one / denom;

                for (size_t d = 0; d < D; ++d) {
                    const nb::BFloat16 normalized_val = x(b, s, d) * inv_rms;
                    output(b, s, d) = normalized_val * weight->data(0, 0, d);
                }
            }
        }

        return Variable::create(output, "rms_norm_output");
    }
    void loadWeights(std::istream& file, const MetadataMap& metadata)
    {
        auto it = metadata.find("weight");
        if (it == metadata.end()) {
            std::cerr << "[RMSNorm] weight 메타데이터가 없어 로딩을 건너뜁니다.\n";
            return;
        }

        const auto& info = it->second;
        load_tensor_data_checked("RMSNorm.weight", weight->data, file, info);
    }
};

} // namespace bs
