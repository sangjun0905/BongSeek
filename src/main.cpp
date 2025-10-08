#include <iostream>
#include "Model.hpp"


int main()
{
    //토큰화
    //makemodel에서 모델 초기 설정 & 가중치 읽어옴
    //greedydecode -> forward 연산 수행 후 가장 큰 값 뽑은 tensor
    //다시 토크나이저로 단어로 만들기

    BongSeek model = makeModel(/*arguments*/);
    Tensor x;
    MatrixXd test(1, 1);
    test << 3;
    x.push_back(test);
    cout << "\n---forward---\n";
    model.forward(x);
    std::cout << "\nHi Hello World!\n";

}