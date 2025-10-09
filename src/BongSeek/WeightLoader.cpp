#include "BongSeek/WeightLoader.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <iterator>
#include <numeric>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "NumBong/BFloat16.hpp"

namespace {

std::string to_lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

std::size_t dtype_to_size(const std::string& dtype) {
    const std::string key = to_lower(dtype);
    if (key == "f32" || key == "float32") return 4;
    if (key == "bf16" || key == "bfloat16") return 2;
    if (key == "f16" || key == "float16") return 2;
    if (key == "i32" || key == "int32") return 4;
    if (key == "i16" || key == "int16") return 2;
    if (key == "i8" || key == "int8") return 1;
    if (key == "u8" || key == "uint8") return 1;
    return 0;
}

std::vector<float> read_bfloat16(std::ifstream& stream, std::size_t count) {
    std::vector<float> out;
    out.reserve(count);
    std::vector<std::uint8_t> buffer(count * sizeof(nb::BFloat16));
    stream.read(reinterpret_cast<char*>(buffer.data()),
                static_cast<std::streamsize>(buffer.size()));
    const std::size_t bytes_read = static_cast<std::size_t>(stream.gcount());
    if (bytes_read < buffer.size()) {
        buffer.resize(bytes_read);
    }
    const std::size_t value_count = buffer.size() / sizeof(nb::BFloat16);
    for (std::size_t i = 0; i < value_count; ++i) {
        const std::uint16_t raw =
            static_cast<std::uint16_t>(buffer[2 * i]) |
            static_cast<std::uint16_t>(buffer[2 * i + 1]) << 8;
        out.push_back(static_cast<float>(nb::BFloat16::from_bits(raw)));
    }
    return out;
}

std::vector<float> read_float32(std::ifstream& stream, std::size_t count) {
    std::vector<float> out(count);
    stream.read(reinterpret_cast<char*>(out.data()),
                static_cast<std::streamsize>(count * sizeof(float)));
    const std::size_t values_read = static_cast<std::size_t>(stream.gcount() / sizeof(float));
    out.resize(values_read);
    return out;
}

template<typename Int>
std::vector<float> read_integer(std::ifstream& stream, std::size_t count) {
    std::vector<Int> buffer(count);
    stream.read(reinterpret_cast<char*>(buffer.data()),
                static_cast<std::streamsize>(count * sizeof(Int)));
    const std::size_t values_read = static_cast<std::size_t>(stream.gcount() / sizeof(Int));
    buffer.resize(values_read);
    std::vector<float> out;
    out.reserve(buffer.size());
    for (const auto value : buffer) {
        out.push_back(static_cast<float>(value));
    }
    return out;
}

} // namespace

using json = nlohmann::json;

std::size_t WeightLoader::Tensor::element_count() const {
    if (shape.empty()) return 1;
    return std::accumulate(shape.begin(), shape.end(), static_cast<std::size_t>(1),
                           std::multiplies<>());
}

std::size_t WeightLoader::Tensor::bytes_per_element() const {
    return dtype_to_size(dtype);
}

std::size_t WeightLoader::Tensor::byte_length() const {
    if (data_end < data_offset) return 0;
    return static_cast<std::size_t>(data_end - data_offset);
}

bool WeightLoader::load(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "[WeightLoader] Failed to open safetensors file: "
                  << path << "\n";
        return false;
    }

    std::uint64_t header_size = 0;
    file.read(reinterpret_cast<char*>(&header_size), sizeof(header_size));
    if (!file) {
        std::cerr << "[WeightLoader] Unable to read safetensors header size from: "
                  << path << "\n";
        return false;
    }

    std::string header_json(static_cast<std::size_t>(header_size), '\0');
    file.read(header_json.data(), static_cast<std::streamsize>(header_json.size()));
    if (!file) {
        std::cerr << "[WeightLoader] Failed to read safetensors JSON header from: "
                  << path << "\n";
        return false;
    }

    json header;
    try {
        header = json::parse(header_json);
    } catch (const std::exception& e) {
        std::cerr << "[WeightLoader] JSON parse error: " << e.what() << "\n";
        return false;
    }

    file_path_ = path;
    header_size_ = header_size;
    tensors_.clear();
    index_.clear();

    for (auto it = header.begin(); it != header.end(); ++it) {
        if (!it->is_object()) {
            continue;
        }

        const std::string& name = it.key();
        if (name == "__metadata__") {
            continue;
        }

        Tensor tensor;
        tensor.name = name;
        tensor.dtype = it->value("dtype", std::string{});

        const auto shape_node = it->find("shape");
        if (shape_node != it->end() && shape_node->is_array()) {
            for (const auto& dim : *shape_node) {
                tensor.shape.push_back(dim.get<std::size_t>());
            }
        }

        const auto offsets_node = it->find("data_offsets");
        if (offsets_node != it->end() && offsets_node->is_array() && offsets_node->size() == 2) {
            tensor.data_offset = offsets_node->at(0).get<std::uint64_t>();
            tensor.data_end = offsets_node->at(1).get<std::uint64_t>();
        } else {
            tensor.data_offset = 0;
            tensor.data_end = 0;
        }

        if (tensor.bytes_per_element() == 0) {
            std::cerr << "[WeightLoader] Warning: unsupported dtype '" << tensor.dtype
                      << "' for tensor " << tensor.name << "\n";
        }

        index_[tensor.name] = tensors_.size();
        tensors_.push_back(std::move(tensor));
    }

    return !tensors_.empty();
}

bool WeightLoader::has(const std::string& name) const {
    return find(name) != nullptr;
}

std::vector<std::size_t> WeightLoader::get_shape(const std::string& name) const {
    const auto* tensor = find(name);
    if (!tensor) return {};
    return tensor->shape;
}

std::vector<float> WeightLoader::get(const std::string& name,
                                     std::size_t max_elements) const {
    const auto* tensor = find(name);
    if (!tensor) return {};
    if (file_path_.empty()) return {};

    const std::size_t bytes_per_element = tensor->bytes_per_element();
    if (bytes_per_element == 0) {
        std::cerr << "[WeightLoader] Unsupported dtype '" << tensor->dtype
                  << "' for tensor " << tensor->name << "\n";
        return {};
    }

    const std::size_t element_count = tensor->element_count();
    if (element_count == 0) return {};

    const std::size_t read_count = (max_elements > 0)
        ? std::min(max_elements, element_count)
        : element_count;
    const std::size_t available_bytes = tensor->byte_length();
    const std::size_t requested_bytes = read_count * bytes_per_element;
    if (requested_bytes > available_bytes) {
        std::cerr << "[WeightLoader] Requested " << requested_bytes
                  << " bytes but tensor '" << tensor->name
                  << "' only stores " << available_bytes << " bytes. Truncating.\n";
    }
    const std::size_t capped_bytes = std::min(requested_bytes, available_bytes);
    const std::size_t capped_count = capped_bytes / bytes_per_element;

    std::ifstream file(file_path_, std::ios::binary);
    if (!file) {
        std::cerr << "[WeightLoader] Failed to reopen safetensors file: "
                  << file_path_ << "\n";
        return {};
    }

    const std::uint64_t absolute_offset =
        static_cast<std::uint64_t>(base_offset()) + tensor->data_offset;
    file.seekg(static_cast<std::streamoff>(absolute_offset));
    if (!file) {
        std::cerr << "[WeightLoader] Failed to seek to tensor '" << tensor->name
                  << "' at offset " << absolute_offset << "\n";
        return {};
    }

    if (capped_count == 0) {
        return {};
    }

    const std::string dtype_lower = to_lower(tensor->dtype);
    if (dtype_lower == "bf16" || dtype_lower == "bfloat16") {
        return read_bfloat16(file, capped_count);
    }
    if (dtype_lower == "f32" || dtype_lower == "float32") {
        return read_float32(file, capped_count);
    }
    if (dtype_lower == "i32" || dtype_lower == "int32") {
        return read_integer<std::int32_t>(file, capped_count);
    }
    if (dtype_lower == "i16" || dtype_lower == "int16") {
        return read_integer<std::int16_t>(file, capped_count);
    }
    if (dtype_lower == "i8" || dtype_lower == "int8") {
        return read_integer<std::int8_t>(file, capped_count);
    }
    if (dtype_lower == "u8" || dtype_lower == "uint8") {
        return read_integer<std::uint8_t>(file, capped_count);
    }

    std::cerr << "[WeightLoader] Unsupported dtype '" << tensor->dtype
              << "' for tensor " << tensor->name << "\n";
    return {};
}

void WeightLoader::print_all_tensors() const {
    std::cout << "  [WeightLoader] Tensors (" << tensors_.size() << ")\n";
    for (const auto& tensor : tensors_) {
        std::cout << "    • " << tensor.name << "  dtype=" << tensor.dtype << "  shape=[";
        for (std::size_t i = 0; i < tensor.shape.size(); ++i) {
            std::cout << tensor.shape[i];
            if (i + 1 < tensor.shape.size()) {
                std::cout << ", ";
            }
        }
        std::cout << "]\n";
    }
}

const WeightLoader::Tensor* WeightLoader::find(const std::string& name) const {
    const auto it = index_.find(name);
    if (it == index_.end()) return nullptr;
    const std::size_t idx = it->second;
    if (idx >= tensors_.size()) return nullptr;
    return &tensors_[idx];
}

std::uint64_t WeightLoader::data_section_offset() const {
    return static_cast<std::uint64_t>(base_offset());
}
