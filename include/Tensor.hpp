#pragma once

#include <vector>


class Tensor {
public:
    using index_t = int;

    Tensor();
    explicit Tensor(std::initializer_list<index_t>);
    explicit Tensor(const std::vector<index_t>&);

    index_t size() const;
    index_t ndim() const;
    const std::vector<index_t>& shape() const;
}