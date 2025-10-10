#include "BongSeek/Runtime.hpp"
#include "BongTorch/Core.hpp"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <filesystem>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct CliOptions {
    bongseek::RuntimeOptions runtime;
};

std::optional<std::size_t> parse_layers_argument(const char* value) {
    if (value == nullptr || *value == '\0') {
        return std::nullopt;
    }
    try {
        return std::stoul(value);
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

fs::path find_model_directory(const std::optional<fs::path>& executable_path) {
    std::vector<fs::path> hints;
    hints.push_back(fs::current_path());
    if (executable_path && !executable_path->empty()) {
        hints.push_back(executable_path->parent_path());
        hints.push_back(executable_path->parent_path().parent_path());
    }

    std::error_code ec;
    for (auto hint : hints) {
        if (hint.empty()) {
            continue;
        }
        hint = fs::absolute(hint, ec);
        if (ec) {
            ec.clear();
            continue;
        }

        auto current = hint;
        while (!current.empty()) {
            const auto candidate = current / "model";
            if (fs::exists(candidate, ec) && !ec) {
                return candidate;
            }
            const auto parent = current.parent_path();
            if (parent == current) {
                break;
            }
            current = parent;
        }
    }
    return {};
}

CliOptions parse_cli(int argc, char** argv) {
    CliOptions options;
    if (argc > 0 && argv[0]) {
        options.runtime.executable_path = fs::path(argv[0]);
    }

    if (argc > 1 && argv[1] && *argv[1]) {
        options.runtime.weights_path = fs::path(argv[1]);
    }
    if (argc > 2 && argv[2] && *argv[2]) {
        options.runtime.config_path = fs::path(argv[2]);
    }
    options.runtime.layers_to_run = 0;

    for (int index = 3; index < argc; ++index) {
        const char* value = argv[index];
        if (value == nullptr || *value == '\0') {
            continue;
        }
        const std::string_view view(value);
        const bool looks_like_path = view.find('/') != std::string_view::npos ||
                                     view.find('\\') != std::string_view::npos ||
                                     view.find('.') != std::string_view::npos;
        if (!options.runtime.tokenizer_path && looks_like_path) {
            options.runtime.tokenizer_path = fs::path(view);
            continue;
        }
        if (const auto layers = parse_layers_argument(value)) {
            options.runtime.layers_to_run = *layers;
            continue;
        }
        if (!options.runtime.tokenizer_path) {
            options.runtime.tokenizer_path = fs::path(view);
        }
    }

    const fs::path model_dir = find_model_directory(options.runtime.executable_path);
    if (!model_dir.empty()) {
        if (!options.runtime.weights_path) {
            const fs::path default_weights = model_dir / "model.safetensors";
            if (fs::exists(default_weights)) {
                options.runtime.weights_path = default_weights;
            }
        }
        if (!options.runtime.config_path) {
            const fs::path default_config = model_dir / "config.json";
            if (fs::exists(default_config)) {
                options.runtime.config_path = default_config;
            }
        }
        if (!options.runtime.tokenizer_path) {
            const std::vector<std::string> tokenizer_candidates = {
                "tokenizer.model",
                "spiece.model",
                "tokenizer.spm",
                "tokenizer.json"
            };
            for (const auto& candidate : tokenizer_candidates) {
                const fs::path candidate_path = model_dir / candidate;
                if (fs::exists(candidate_path)) {
                    options.runtime.tokenizer_path = candidate_path;
                    break;
                }
            }
        }
    }

    return options;
}

int select_next_token(const Tensor& logits, const Tokenizer& tokenizer) {
    const auto shape = logits.getShape();
    if (shape[0] == 0 || shape[1] == 0 || shape[2] == 0) {
        return tokenizer.pad_token_id();
    }

    const std::size_t last_index = shape[1] - 1;
    float best_score = std::numeric_limits<float>::lowest();
    int best_id = tokenizer.pad_token_id() >= 0 ? tokenizer.pad_token_id() : 0;
    bool found = false;

    for (std::size_t vocab_id = 0; vocab_id < shape[2]; ++vocab_id) {
        const int candidate = static_cast<int>(vocab_id);
        if (tokenizer.is_special_id(candidate)) {
            continue;
        }
        const float score = logits(0, last_index, vocab_id).to_float();
        if (!found || score > best_score) {
            best_score = score;
            best_id = candidate;
            found = true;
        }
    }

    if (!found) {
        // fallback: allow special tokens if we filtered everything out
        best_score = std::numeric_limits<float>::lowest();
        for (std::size_t vocab_id = 0; vocab_id < shape[2]; ++vocab_id) {
            const float score = logits(0, last_index, vocab_id).to_float();
            if (score > best_score) {
                best_score = score;
                best_id = static_cast<int>(vocab_id);
            }
        }
    }

    return best_id;
}

void trim_context(std::vector<int>& tokens, std::size_t max_tokens, int bos_id) {
    if (tokens.size() <= max_tokens) {
        return;
    }

    std::vector<int> trimmed;
    trimmed.reserve(max_tokens);
    if (bos_id >= 0) {
        trimmed.push_back(bos_id);
    }
    if (trimmed.size() >= max_tokens) {
        tokens.swap(trimmed);
        return;
    }

    const std::size_t keep = std::min(max_tokens - trimmed.size(), tokens.size());
    const std::size_t start = tokens.size() - keep;
    trimmed.insert(trimmed.end(), tokens.begin() + static_cast<std::ptrdiff_t>(start), tokens.end());
    tokens.swap(trimmed);
}

void run_chatbot(bongseek::RuntimeContext& ctx) {
    std::vector<int> context_tokens;
    if (ctx.tokenizer.bos_token_id() >= 0) {
        context_tokens.push_back(ctx.tokenizer.bos_token_id());
    }

    const std::vector<int> newline_tokens = ctx.tokenizer.encode("\n");
    const std::size_t configured_context_limit =
        ctx.config.max_position_embeddings > 0
            ? static_cast<std::size_t>(ctx.config.max_position_embeddings)
            : 0;
    const std::size_t max_context_tokens =
        configured_context_limit > 0 ? configured_context_limit : static_cast<std::size_t>(1024);

    std::cout << "[Chatbot] Ready. Type /exit to quit, /reset to clear context.\n";

    std::string line;
    while (true) {
        std::cout << "You> ";
        if (!std::getline(std::cin, line)) {
            std::cout << '\n';
            break;
        }

        if (line == "/exit" || line == "/quit") {
            std::cout << "[Chatbot] Goodbye!\n";
            break;
        }
        if (line == "/reset") {
            context_tokens.clear();
            if (ctx.tokenizer.bos_token_id() >= 0) {
                context_tokens.push_back(ctx.tokenizer.bos_token_id());
            }
            std::cout << "[Chatbot] Context cleared.\n";
            continue;
        }

        auto user_tokens = ctx.tokenizer.encode(line);
        if (user_tokens.empty()) {
            std::cout << "[Chatbot] Unable to encode input.\n";
            continue;
        }

        context_tokens.insert(context_tokens.end(), user_tokens.begin(), user_tokens.end());
        context_tokens.insert(context_tokens.end(), newline_tokens.begin(), newline_tokens.end());
        trim_context(context_tokens, max_context_tokens, ctx.tokenizer.bos_token_id());

        std::vector<int> reply_tokens;
        const std::size_t max_new_tokens = 64;
        bool has_visible_output = false;

        for (std::size_t step = 0; step < max_new_tokens; ++step) {
            trim_context(context_tokens, max_context_tokens, ctx.tokenizer.bos_token_id());
            Tensor logits = bongseek::forward_tokens(ctx, context_tokens);
            const int next_id = select_next_token(logits, ctx.tokenizer);
            context_tokens.push_back(next_id);
            reply_tokens.push_back(next_id);
            trim_context(context_tokens, max_context_tokens, ctx.tokenizer.bos_token_id());

            const std::string token_text = ctx.tokenizer.decode({next_id});
            if (std::any_of(token_text.begin(), token_text.end(), [](char ch) {
                    return !std::isspace(static_cast<unsigned char>(ch));
                })) {
                has_visible_output = true;
            }
            if (ctx.tokenizer.is_special_id(next_id)) {
                break;
            }
            if (ctx.tokenizer.eos_token_id() >= 0 &&
                next_id == ctx.tokenizer.eos_token_id()) {
                break;
            }
            if (!token_text.empty() && token_text.find('\n') != std::string::npos) {
                if (has_visible_output) {
                    break;
                }
                continue;
            }
        }

        context_tokens.insert(context_tokens.end(), newline_tokens.begin(), newline_tokens.end());
        trim_context(context_tokens, max_context_tokens, ctx.tokenizer.bos_token_id());

        std::string reply = ctx.tokenizer.decode(reply_tokens);
        while (!reply.empty() && (reply.back() == '\n' || reply.back() == '\r')) {
            reply.pop_back();
        }

        if (reply.empty()) {
            std::cout << "Bot> [no output]\n";
        } else {
            std::cout << "Bot> " << reply << "\n";
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        CliOptions options = parse_cli(argc, argv);
        auto runtime = bongseek::initialize_runtime(options.runtime);
        run_chatbot(runtime);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[Main] Unhandled exception: " << e.what() << '\n';
    } catch (...) {
        std::cerr << "[Main] Unhandled unknown exception" << '\n';
    }
    return 1;
}
