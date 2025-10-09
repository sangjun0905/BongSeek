#pragma once

#include <cmath>
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
        return bits_to_float(bits);
    }

    static float to_float(storage_type raw) {
        return bits_to_float(raw);
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

    BFloat16 operator-() const {
        return from_bits(static_cast<storage_type>(bits ^ 0x8000u));
    }

    friend bool operator==(const BFloat16& lhs, const BFloat16& rhs) {
        return lhs.bits == rhs.bits;
    }

    friend bool operator!=(const BFloat16& lhs, const BFloat16& rhs) {
        return !(lhs == rhs);
    }

    friend bool operator<(const BFloat16& lhs, const BFloat16& rhs) {
        return lhs.to_float() < rhs.to_float();
    }

    friend bool operator<=(const BFloat16& lhs, const BFloat16& rhs) {
        return lhs.to_float() <= rhs.to_float();
    }

    friend bool operator>(const BFloat16& lhs, const BFloat16& rhs) {
        return lhs.to_float() > rhs.to_float();
    }

    friend bool operator>=(const BFloat16& lhs, const BFloat16& rhs) {
        return lhs.to_float() >= rhs.to_float();
    }

private:
    void set_from_float(float value) {
        bits = float_to_bits(value);
    }

    static storage_type float_to_bits(float value) {
        std::uint32_t full{};
        std::memcpy(&full, &value, sizeof(full));

        constexpr std::uint32_t exponent_mask = 0x7F800000u;
        constexpr std::uint32_t mantissa_mask = 0x007FFFFFu;

        const std::uint32_t exponent = full & exponent_mask;
        const std::uint32_t mantissa = full & mantissa_mask;

        if (exponent == exponent_mask) {
            storage_type upper = static_cast<storage_type>(full >> 16);
            if (mantissa != 0) {
                // Quiet-NaN payload: ensure MSB of mantissa is set
                upper |= static_cast<storage_type>(0x0040u);
            }
            return upper;
        }

        const std::uint32_t lsb = (full >> 16) & 1u;
        const std::uint32_t rounding_bias = 0x00007FFFu + lsb;
        full += rounding_bias;

        return static_cast<storage_type>(full >> 16);
    }

    static float bits_to_float(storage_type raw) {
        std::uint32_t full = static_cast<std::uint32_t>(raw) << 16;
        float value{};
        std::memcpy(&value, &full, sizeof(value));
        return value;
    }
};

static_assert(sizeof(BFloat16) == 2, "BFloat16 must occupy 16 bits");
static_assert(std::is_trivially_copyable_v<BFloat16>, "BFloat16 must be trivially copyable");

inline BFloat16 bfloat16_sqrt(const BFloat16& value) {
    return BFloat16(std::sqrt(static_cast<float>(value)));
}

inline BFloat16 bfloat16_rsqrt(const BFloat16& value) {
    return BFloat16(1.0f / std::sqrt(static_cast<float>(value)));
}

inline BFloat16 bfloat16_exp(const BFloat16& value) {
    return BFloat16(std::exp(static_cast<float>(value)));
}

} // namespace nb
