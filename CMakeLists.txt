cmake_minimum_required(VERSION 3.16)

project(BongSeek LANGUAGES CXX)

# Attempt to locate local vcpkg-installed dependencies automatically.
set(_local_vcpkg_root "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg/installed")
if (EXISTS "${_local_vcpkg_root}")
    set(_candidate_triplets)
    file(GLOB _triplet_dirs RELATIVE "${_local_vcpkg_root}" "${_local_vcpkg_root}/*")
    foreach(_triplet ${_triplet_dirs})
        if (IS_DIRECTORY "${_local_vcpkg_root}/${_triplet}")
            list(APPEND _candidate_triplets "${_triplet}")
        endif()
    endforeach()
    # Prefer a triplet that matches the host architecture
    set(_selected_triplet "")
    foreach(_triplet ${_candidate_triplets})
        if (CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin" AND _triplet MATCHES "arm64-osx")
            set(_selected_triplet "${_triplet}")
            break()
        elseif (CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux" AND _triplet MATCHES "x64-linux")
            set(_selected_triplet "${_triplet}")
            break()
        elseif (CMAKE_HOST_SYSTEM_NAME STREQUAL "Windows" AND _triplet MATCHES "x64-windows")
            set(_selected_triplet "${_triplet}")
            break()
        endif()
    endforeach()
    if (_selected_triplet STREQUAL "" AND _candidate_triplets)
        list(GET _candidate_triplets 0 _selected_triplet)
    endif()
    if (NOT _selected_triplet STREQUAL "")
        set(_vcpkg_triplet_dir "${_local_vcpkg_root}/${_selected_triplet}")
        list(APPEND CMAKE_PREFIX_PATH "${_vcpkg_triplet_dir}")
        list(APPEND CMAKE_PREFIX_PATH "${_vcpkg_triplet_dir}/share")
        if (NOT DEFINED ENV{PKG_CONFIG_PATH})
            set(ENV{PKG_CONFIG_PATH} "${_vcpkg_triplet_dir}/lib/pkgconfig")
        else()
            set(ENV{PKG_CONFIG_PATH} "$ENV{PKG_CONFIG_PATH}:${_vcpkg_triplet_dir}/lib/pkgconfig")
        endif()
    endif()
endif()

find_package(Threads REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(SENTENCEPIECE REQUIRED IMPORTED_TARGET sentencepiece)

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
    src/BongSeek/ModelConfig.cpp
    src/BongSeek/Tokenizer.cpp
    src/BongSeek/Runtime.cpp
    src/BongSeek/WeightLoader.cpp
    src/BongSeek/Model.cpp
)


add_executable(BongSeek ${BONGSEEK_SOURCES})
if (SENTENCEPIECE_LIBRARY_DIRS)
    target_link_directories(BongSeek PRIVATE ${SENTENCEPIECE_LIBRARY_DIRS})
endif()

if (TARGET PkgConfig::sentencepiece)
    target_link_libraries(BongSeek PRIVATE BongTorch nlohmann_json::nlohmann_json PkgConfig::sentencepiece)
else()
    target_include_directories(BongSeek PRIVATE ${SENTENCEPIECE_INCLUDE_DIRS})
    target_link_libraries(BongSeek PRIVATE BongTorch nlohmann_json::nlohmann_json ${SENTENCEPIECE_LIBRARIES})
endif()
