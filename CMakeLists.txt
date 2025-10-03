cmake_minimum_required(VERSION 3.16)

project(BongSeek CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SRC_FILES
    src/main.cpp    
)

add_executable(BongSeek ${SRC_FILES})

target_include_directories(BongSeek PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)