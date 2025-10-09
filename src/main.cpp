#include "bongseek/Model.hpp"
#include <iostream>
#include <fstream>
#include "bongseek/WeightLoader.hpp"
#include "bongseek/Config.hpp"

using namespace std;

int main() {
    
    WeightLoader loader;

    string filename ="../model/model.safetensors";

    istream file(filename);
    loader.load(filename);
    
    MetadataMap = loader.get_tensor_map();

    Config config;
    Model model(config);

    model.load_weights(file, metadata);

    

    return 0;
}
