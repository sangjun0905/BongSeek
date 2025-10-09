cmake_minimum_required(VERSION 3.16)

project(BongSeek LANGUAGES CXX)

find_package(Threads REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

add_library(NumBong INTERFACE)
target_include_directories(NumBong INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(NumBong INTERFACE cxx_std_20)
target_link_libraries(NumBong INTERFACE Threads::Threads)

add_library(BongTorch INTERFACE)
target_include_directories(BongTorch INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(BongTorch INTERFACE NumBong)

add_library(bongseek INTERFACE)
target_include_directories(bongseek INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)


set(BONGSEEK_SOURCES
    src/main.cpp
    src/BongSeek/AssetLocator.cpp
    src/BongSeek/ModelConfig.cpp
    src/BongSeek/ModelRuntime.cpp
    src/BongSeek/Tokenizer.cpp
    src/BongSeek/TransformerModel.cpp
    src/BongSeek/WeightLoader.cpp
    src/BongSeek/LLM.cpp
    src/BongSeek/WeightLoader.cpp
    src/BongSeek/Model.cpp
)


add_executable(BongSeek ${BONGSEEK_SOURCES})
target_link_libraries(BongSeek PRIVATE BongTorch nlohmann_json::nlohmann_json)
