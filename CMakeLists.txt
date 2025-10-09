cmake_minimum_required(VERSION 3.16)

project(BongSeek LANGUAGES CXX)

add_library(NumBong INTERFACE)
target_include_directories(NumBong INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(NumBong INTERFACE cxx_std_20)

add_library(BongTorch INTERFACE)
target_include_directories(BongTorch INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(BongTorch INTERFACE NumBong)

add_executable(BongSeek src/main.cpp)
target_sources(BongSeek PRIVATE src/model/ModelConfig.cpp)
target_link_libraries(BongSeek PRIVATE BongTorch)
target_include_directories(BongSeek PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vcpkg/installed/x64-windows/include)
