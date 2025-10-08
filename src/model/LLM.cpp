#include "LLM.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <random>

// softmax (온도 적용)
static std::vector<float> softmax(const std::vector<float>& logits, float temperature = 1.0f) {
    std::vector<float> probs(logits.size());
    float max_logit = *std::max_element(logits.begin(), logits.end());

    float sum = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i) {
        float val = std::exp((logits[i] - max_logit) / temperature);
        probs[i] = val;
        sum += val;
    }
    for (float& p : probs) p /= sum;
    return probs;
}

// 확률 기반 샘플링 (Torch 팀 샘플링 대체용)
static int sample_from_distribution(const std::vector<float>& probs) {
    static std::mt19937 rng(std::random_device{}());
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    return dist(rng);
}

// ------------------------------------------------------
// LLM::generate()
// ------------------------------------------------------
std::string LLM::generate(const std::string& prompt, int max_tokens) {
    std::cout << "\n[LLM] 텍스트 생성 시작\n";
    auto start_time = std::chrono::high_resolution_clock::now();

    // (1) 입력을 토큰으로 변환
    auto input_ids = tokenizer.encode(prompt);
    std::cout << "[Tokenizer] 입력 토큰 개수: " << input_ids.size() << "\n";

    // (2) 토큰 생성 루프
    for (int step = 0; step < max_tokens; ++step) {
        // forward (Torch 팀 모델 호출)
        auto logits = model.forward(input_ids);

        // softmax → 확률 분포
        auto probs = softmax(logits, 0.8f);

        // 샘플링 (가장 단순한 버전)
        int next_token = sample_from_distribution(probs);

        input_ids.push_back(next_token);

        // 중간 출력 로그
        std::cout << "[Step " << std::setw(2) << step+1 << "] Token=" << next_token << "\n";

        // EOS 토큰이면 중단
        if (next_token == config.eos_token_id) {
            std::cout << "[LLM] <eos> 토큰 도달 → 종료\n";
            break;
        }
    }

    // (3) 토큰 → 문자열 변환
    std::string output = tokenizer.decode(input_ids);

    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end_time - start_time).count();
    std::cout << "[LLM] 텍스트 생성 완료 (" << elapsed << "초)\n";

    return output;
}
