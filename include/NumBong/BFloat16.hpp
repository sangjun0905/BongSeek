#pragma once

#include <cstdint>
#include <cstring>
#include <type_traits>

namespace nb {

struct alignas(2) BFloat16 {
    using storage_type = std::uint16_t;
    storage_type bits{};

    BFloat16() = default;

    explicit BFloat16(float value) {
        set_from_float(value);
    }

    explicit BFloat16(double value) : BFloat16(static_cast<float>(value)) {}

    template<typename Int, std::enable_if_t<std::is_integral_v<Int>, int> = 0>
    explicit BFloat16(Int value) : BFloat16(static_cast<float>(value)) {}

    static BFloat16 from_bits(storage_type raw) {
        BFloat16 out;
        out.bits = raw;
        return out;
    }

    storage_type raw_bits() const {
        return bits;
    }

    float to_float() const {
        std::uint32_t full = static_cast<std::uint32_t>(bits) << 16;
        float value{};
        std::memcpy(&value, &full, sizeof(value));
        return value;
    }

    explicit operator float() const { return to_float(); }
    explicit operator double() const { return static_cast<double>(to_float()); }

    BFloat16& operator+=(const BFloat16& other) {
        *this = *this + other;
        return *this;
    }

    BFloat16& operator-=(const BFloat16& other) {
        *this = *this - other;
        return *this;
    }

    BFloat16& operator*=(const BFloat16& other) {
        *this = *this * other;
        return *this;
    }

    BFloat16& operator/=(const BFloat16& other) {
        *this = *this / other;
        return *this;
    }

    friend BFloat16 operator+(const BFloat16& lhs, const BFloat16& rhs) {
        return BFloat16(lhs.to_float() + rhs.to_float());
    }

    friend BFloat16 operator-(const BFloat16& lhs, const BFloat16& rhs) {
        return BFloat16(lhs.to_float() - rhs.to_float());
    }

    friend BFloat16 operator*(const BFloat16& lhs, const BFloat16& rhs) {
        return BFloat16(lhs.to_float() * rhs.to_float());
    }

    friend BFloat16 operator/(const BFloat16& lhs, const BFloat16& rhs) {
        return BFloat16(lhs.to_float() / rhs.to_float());
    }

    friend bool operator==(const BFloat16& lhs, const BFloat16& rhs) {
        return lhs.bits == rhs.bits;
    }

    friend bool operator!=(const BFloat16& lhs, const BFloat16& rhs) {
        return !(lhs == rhs);
    }

private:
    void set_from_float(float value) {
        std::uint32_t full{};
        std::memcpy(&full, &value, sizeof(value));
        bits = static_cast<storage_type>(full >> 16);
    }
};

static_assert(sizeof(BFloat16) == 2, "BFloat16 must occupy 16 bits");
static_assert(std::is_trivially_copyable_v<BFloat16>, "BFloat16 must be trivially copyable");

} // namespace nb