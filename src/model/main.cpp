#include <iostream>
#include "LLM.hpp"

int main() {
    std::cout << "=== LiquidAI LFM2-2.6B (C++ Interface Test) ===\n";

    LLM llm;
    if (!llm.load("LiquidAI/LFM2-2.6B")) {
        std::cerr << "모델 로드 실패!" << std::endl;
        return 1;
    }

    std::string prompt;
    std::cout << "\n[Prompt 입력] ";
    std::getline(std::cin, prompt);

    std::string output = llm.generate(prompt, 30);

    std::cout << "\n=== 결과 ===\n" << output << std::endl;
    return 0;
}
