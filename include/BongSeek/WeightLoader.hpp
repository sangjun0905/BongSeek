#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

/**
 * Lightweight reader for HuggingFace safetensors files.
 *
 * The loader keeps tensor metadata in memory and reads tensor data
 * lazily on demand.  Callers can request either the raw tensor metadata
 * through `tensors()` or materialize tensor values with `get()`.
 *
 * The safetensors container encodes all tensors in a flat binary blob:
 *   [8 byte little-endian header size][JSON header][tensor bytes...]
 *
 * The JSON header provides dtype, shape, and data offsets for each tensor.
 */
class WeightLoader {
public:
    struct Tensor {
        std::string name;
        std::string dtype;
        std::vector<std::size_t> shape;
        std::uint64_t data_offset = 0; ///< offset relative to the data section
        std::uint64_t data_end = 0;    ///< offset relative to the data section

        [[nodiscard]] std::size_t element_count() const;
        [[nodiscard]] std::size_t bytes_per_element() const;
        [[nodiscard]] std::size_t byte_length() const;
    };

    using TensorList = std::vector<Tensor>;

    WeightLoader() = default;

    /**
     * Parse the safetensors file located at `path`.
     *
     * Returns true when the header and metadata are parsed successfully.
     * Metadata stays cached so repeated lookups avoid reparsing the file.
     */
    bool load(const std::filesystem::path& path);

    [[nodiscard]] bool has(const std::string& name) const;
    [[nodiscard]] std::vector<std::size_t> get_shape(const std::string& name) const;

    /**
     * Materialize tensor data as floating-point values.
     *
     * @param name Name of the tensor within the safetensors archive.
     * @param max_elements Optional cap on the number of elements to read.
     *        Pass zero to load the full tensor.  When capped, only the
     *        leading elements are returned (useful for sampling large tensors).
     */
    [[nodiscard]] std::vector<float> get(const std::string& name,
                                         std::size_t max_elements = 0) const;

    /**
     * Return a vector view of the discovered tensor metadata.
     */
    [[nodiscard]] const TensorList& tensors() const { return tensors_; }

    /**
     * Pretty-print the tensor inventory for quick inspection.
     */
    void print_all_tensors() const;

    /**
     * Offset (in bytes) from the start of the file to the data section.
     */
    [[nodiscard]] std::uint64_t data_section_offset() const;

    /**
     * Path of the currently loaded safetensors file.
     */
    [[nodiscard]] const std::filesystem::path& file_path() const { return file_path_; }

private:
    [[nodiscard]] const Tensor* find(const std::string& name) const;
    [[nodiscard]] std::size_t base_offset() const { return header_size_ + sizeof(std::uint64_t); }

    std::filesystem::path file_path_;
    std::uint64_t header_size_ = 0;
    TensorList tensors_;
    std::unordered_map<std::string, std::size_t> index_;
};
