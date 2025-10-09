

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <cstdint>
#include <memory>
#include <stdexcept>
/*
float bfloat16_to_float(uint16_t bf16_val) {
	uint32_t val32 = static_cast<uint32_t>(bf16_val) << 16;
	return *reinterpret_cast<float*>(&val32);
}*/