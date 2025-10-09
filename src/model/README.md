설치할 것 모음
1. C++ sentencepiece -> msys2 mingw64 터미널에서 실행필요
2. json 설치

라이브러리 설치는 msys 터미널에서, 프로그램 빌드는 mingw 터미널에서


cmake .. -G "Ninja" \
  -DCMAKE_C_COMPILER=gcc \
  -DCMAKE_CXX_COMPILER=g++ \
  -DCMAKE_TOOLCHAIN_FILE="/c/working/git_clone/new folder/BongSeek/vcpkg/scripts/buildsystems/vcpkg.cmake" \
  -Dnlohmann_json_DIR="/c/working/git_clone/new folder/BongSeek/vcpkg/installed/x64-windows/share/nlohmann_json"

 cmake --build . --clean-first


 C:\working\git_clone\bongseek_clean> cmake --build build\model --target BongSeekModel
 build\model\BongSeekModel.exe
 chcp 65001