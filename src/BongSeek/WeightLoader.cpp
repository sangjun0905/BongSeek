#include "BongSeek/WeightLoader.hpp"
#include <cstdint>
#include <iostream>
#include "NumBong/BFloat16.hpp"

using json = nlohmann::json;
using namespace std;

// ---- safetensors 헤더 읽기 ----
bool WeightLoader::load(const std::string& path) {
    file_path = path;
    tensor_map.clear();
    file.open(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[WeightLoader] 파일 읽기 실패: " << path << std::endl;
        return false;
    }

    std::uint64_t header_len = 0;
    file.read(reinterpret_cast<char*>(&header_len), sizeof(header_len));
    if (!file) {
        std::cerr << "[WeightLoader] 헤더 길이 읽기 실패\n";
        return false;
    }

    std::string header_str(static_cast<std::size_t>(header_len), '\0');
    if (header_len > 0) {
        file.read(header_str.data(), static_cast<std::streamsize>(header_len));
        if (!file) {
            std::cerr << "[WeightLoader] 헤더 데이터 읽기 실패\n";
            return false;
        }
    }

    json header;
    try {
        header = json::parse(header_str);
    } catch (const std::exception& e) {
        std::cerr << "[WeightLoader] 헤더 파싱 실패: " << e.what() << std::endl;
        return false;
    }

    const std::size_t data_base_offset =
        static_cast<std::size_t>(sizeof(std::uint64_t) + header_len);

    auto register_tensor = [&](const std::string& tensor_name, const json& meta) {
        if (tensor_name.empty() || tensor_name == "__metadata__" || !meta.is_object()) {
            return;
        }

        TensorInfo info;
        info.dtype = meta.value("data_type", meta.value("dtype", std::string{}));
        if (info.dtype.empty()) {
            return;
        }

        if (meta.contains("shape") && meta["shape"].is_array()) {
            for (const auto& d : meta["shape"]) {
                info.shape.push_back(d.get<std::size_t>());
            }
        }

        std::size_t element_size = 0;
        if (info.dtype == "BF16" || info.dtype == "bf16" || info.dtype == "BFloat16") {
            element_size = sizeof(std::uint16_t);
        } else if (info.dtype == "F32" || info.dtype == "f32" || info.dtype == "Float32") {
            element_size = sizeof(float);
        } else {
            std::cerr << "[WeightLoader] 지원되지 않는 dtype(" << info.dtype
                      << ") - " << tensor_name << std::endl;
            return;
        }

        std::size_t element_count = 1;
        for (const auto dim : info.shape) {
            element_count *= dim;
        }
        const std::size_t byte_count = element_count * element_size;

        std::size_t relative_start = 0;
        std::size_t relative_end = 0;
        if (meta.contains("data_offsets") && meta["data_offsets"].is_array() && meta["data_offsets"].size() == 2) {
            relative_start = meta["data_offsets"][0].get<std::size_t>();
            relative_end = meta["data_offsets"][1].get<std::size_t>();
        } else {
            if (meta.contains("offset")) {
                relative_start = meta["offset"].get<std::size_t>();
            } else if (meta.contains("offset_start")) {
                relative_start = meta["offset_start"].get<std::size_t>();
            } else {
                std::cerr << "[WeightLoader] offset 정보 누락: " << tensor_name << std::endl;
                return;
            }
            relative_end = relative_start + byte_count;
        }

        if (relative_end < relative_start) {
            std::cerr << "[WeightLoader] 잘못된 offset 범위: " << tensor_name << std::endl;
            return;
        }

        const std::size_t span = relative_end - relative_start;
        if (byte_count != 0 && span != byte_count) {
            std::cerr << "[WeightLoader] 크기 불일치: " << tensor_name
                      << " | metadata=" << span << " bytes, expected=" << byte_count << " bytes"
                      << std::endl;
        }

        info.offset_start = data_base_offset + relative_start;
        info.offset_end = data_base_offset + relative_end;

        tensor_map[tensor_name] = info;
    };

    if (header.is_array()) {
        for (const auto& entry : header) {
            const std::string name = entry.value("layername", std::string{});
            register_tensor(name.empty() ? entry.value("name", std::string{}) : name, entry);
        }
    } else if (header.is_object()) {
        for (auto& [name, meta] : header.items()) {
            std::string tensor_name = meta.value("layername", name);
            register_tensor(tensor_name, meta);
        }
    } else {
        std::cerr << "[WeightLoader] 지원되지 않는 헤더 형식입니다.\n";
        return false;
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
    if (info.offset_end < info.offset_start) {
        std::cerr << "[WeightLoader] 잘못된 오프셋 범위: " << tensor_name << std::endl;
        return {};
    }
    const std::size_t bytes = info.offset_end - info.offset_start;

    std::vector<float> data; // 바이너리 데이터 저장된 벡터
    file.seekg(static_cast<std::streamoff>(info.offset_start), std::ios::beg);
    if (!file) {
        std::cerr << "[WeightLoader] 파일 seek 실패: " << tensor_name << std::endl;
        return {};
    }

    if (info.dtype == "F32" || info.dtype == "f32" || info.dtype == "Float32") {
        size_t count = bytes / sizeof(float);
        data.resize(count);
        file.read(reinterpret_cast<char*>(data.data()), bytes);
    } else if (info.dtype == "BF16" || info.dtype == "bf16" || info.dtype == "BFloat16") {
        size_t count = bytes / 2;
        data.resize(count);
        std::vector<uint16_t> tmp(count);
        file.read(reinterpret_cast<char*>(tmp.data()), bytes);
        for (size_t i = 0; i < count; ++i) {
            data[i] = nb::BFloat16::to_float(tmp[i]);
        }
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
        std::cout << " • " << name
                  << " | dtype=" << info.dtype
                  << " | shape=(";
        for (size_t i = 0; i < info.shape.size(); ++i) {
            std::cout << info.shape[i];
            if (i + 1 < info.shape.size()) std::cout << ",";
        }
        std::cout << ")\n";

    }
}

bool WeightLoader::has(const std::string& tensor_name) const {
    return tensor_map.find(tensor_name) != tensor_map.end();
}

size_t WeightLoader::tensor_count() const {
    return tensor_map.size();
}
