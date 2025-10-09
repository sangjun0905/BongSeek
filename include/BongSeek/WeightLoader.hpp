#pragma once
#include <string>
#include <unordered_map>
#include <vector>
#include <fstream>
#include <nlohmann/json.hpp>
#include "../BongTorch/Core.hpp"


/*
struct TensorInfo {
    std::string dtype;
    std::vector<size_t> shape;
    size_t offset_start;
    size_t offset_end;
};*/

class WeightLoader {
private:
    std::string file_path;
    std::ifstream file;
    std::unordered_map<std::string, TensorInfo> tensor_map;

public:
    bool load(const std::string& path);
    std::vector<float> get(const std::string& tensor_name);
    std::vector<size_t> get_shape(const std::string& tensor_name);
    void print_all_tensors(size_t max_count = 20);
    bool has(const std::string& tensor_name) const;
    size_t tensor_count() const;
    std::unordered_map<std::string, TensorInfo> get_tensor_map() const { return tensor_map; }
    ~WeightLoader() { if (file.is_open()) file.close(); }
};
