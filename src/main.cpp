#include <filesystem>
#include <iostream>
#include "model/ModelConfig.hpp"

int main() {
    try {
        const std::filesystem::path source_dir = std::filesystem::path(__FILE__).parent_path();
        const std::filesystem::path model_dir = source_dir / "model" / "sample_data" / "test_model";

        ModelConfig config(model_dir);
        std::cout << "[main] model_type: " << config.model_type << '\n';
        std::cout << "[main] num_hidden_layers: " << config.num_hidden_layers << '\n';
        std::cout << "[main] hidden_size: " << config.hidden_size << '\n';
    } catch (const std::exception& e) {
        std::cerr << "[main] 에러 발생: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
