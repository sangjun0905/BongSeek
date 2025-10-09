#pragma once
#include <vector>
#include <iostream>
// 필요한 헤더들...

namespace nb {   // ✅ 여기에 네임스페이스 nb 정의

// 여기에 모든 Numbong 관련 코드 넣기
// 예시:
class Array {
public:
    Array() = default;
    // ...
};

Array array(std::initializer_list<double> vals) {
    // ...
    return Array();
}

Array ones_like(const Array& x) {
    // ...
    return Array();
}

// 기타 유틸 함수들
// sum_to, reshape, pow, transpose 등

} // namespace nb