#include "WeightLoader.hpp"
#include <iostream>
#include <cmath>
#include <cstring>

using json = nlohmann::json;

// ---- BF16 → float 변환 함수 ----
inline float bf16_to_float(uint16_t val) {
    uint32_t tmp = ((uint32_t)val) << 16;
    float result;
    std::memcpy(&result, &tmp, sizeof(float));
    return result;
}

// ---- safetensors 헤더 읽기 ----
bool WeightLoader::load(const std::string& path) {
    file_path = path;
    file.open(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[WeightLoader] 파일 열기 실패: " << path << std::endl;
        return false;
    }

    uint32_t header_len = 0;
    file.read(reinterpret_cast<char*>(&header_len), 4);
    std::string header_str(header_len, '\0');
    file.read(&header_str[0], header_len);

    json header = json::parse(header_str);

    for (auto& [name, meta] : header.items()) {
        TensorInfo info;
        info.dtype = meta["dtype"].get<std::string>();
        for (auto& d : meta["shape"]) info.shape.push_back(d.get<size_t>());
        info.offset_start = meta["data_offsets"][0].get<size_t>();
        info.offset_end   = meta["data_offsets"][1].get<size_t>();
        tensor_map[name] = info;
    }

    std::cout << "[WeightLoader] " << tensor_map.size() << "개의 텐서 인덱싱 완료." << std::endl;
    return true;
}

// ---- 개별 텐서 읽기 ----
std::vector<float> WeightLoader::get(const std::string& tensor_name) {
    if (!tensor_map.count(tensor_name)) {
        std::cerr << "[WeightLoader] 존재하지 않는 텐서: " << tensor_name << std::endl;
        return {};
    }

    const auto& info = tensor_map[tensor_name];
    size_t bytes = info.offset_end - info.offset_start;

    std::vector<float> data;
    file.seekg(4 + info.offset_start, std::ios::beg);

    if (info.dtype == "F32") {
        size_t count = bytes / sizeof(float);
        data.resize(count);
        file.read(reinterpret_cast<char*>(data.data()), bytes);
    } else if (info.dtype == "BF16") {
        size_t count = bytes / 2;
        data.resize(count);
        std::vector<uint16_t> tmp(count);
        file.read(reinterpret_cast<char*>(tmp.data()), bytes);
        for (size_t i = 0; i < count; ++i)
            data[i] = bf16_to_float(tmp[i]);
    } else {
        std::cerr << "[WeightLoader] 지원되지 않는 dtype: " << info.dtype << std::endl;
        return {};
    }

    return data;
}

// ---- shape 조회 ----
std::vector<size_t> WeightLoader::get_shape(const std::string& tensor_name) {
    if (!tensor_map.count(tensor_name)) return {};
    return tensor_map[tensor_name].shape;
}

// ---- 텐서 목록 출력 ----
void WeightLoader::print_all_tensors(size_t max_count) {
    size_t count = 0;
    for (auto& [name, info] : tensor_map) {
        std::cout << "📦 " << name
                  << " | dtype=" << info.dtype
                  << " | shape=(";
        for (size_t i = 0; i < info.shape.size(); ++i) {
            std::cout << info.shape[i];
            if (i < info.shape.size() - 1) std::cout << ",";
        }
        std::cout << ")\n";

        if (++count >= max_count) {
            std::cout << "… (" << tensor_map.size() - count
                      << " more tensors)\n";
            break;
        }
    }
}
