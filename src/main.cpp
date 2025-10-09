#include "bongseek/Model.hpp"
#include "NumBong/Tensor.hpp"
#include "NumBong/BFloat16.hpp"
#include "BongTorch/Core.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include "bongseek/WeightLoader.hpp"
#include "bongseek/Config.hpp"

using namespace std;

int main() {
    
    WeightLoader loader;

    string filename ="../model/model.safetensors";
    ifstream file(filename, std::ios::binary);
    loader.load(filename);
    
    MetadataMap metadata;

    metadata = loader.get_tensor_map();

    Config config;
    Model model(config);
    model.load_weights(file, metadata);

    nb::Tensor<nb::BFloat16, 3> a(1, 10, 1);
    a.fill(nb::BFloat16(0));                // or assign token indices
    auto x = bs::Variable::create(a);



    model.forward(x);
    
    

    return 0;
}

