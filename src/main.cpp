#include "bongseek/Model.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include "bongseek/WeightLoader.hpp"
#include "bongseek/Config.hpp"

using namespace std;

int main() {
    
    WeightLoader loader;

    string filename ="../model/model.safetensors";
    cout << "test1 "<<endl;
    ifstream file(filename, std::ios::binary);
    cout << "test2 "<<endl;
    loader.load(filename);
    cout << "test3 "<<endl;
    
    MetadataMap metadata;

    metadata = loader.get_tensor_map();

    cout << "test4 "<<endl;
    
    Config config;
    Model model(config);

    cout << "test5 "<<endl;

    loader.print_all_tensors();
    model.load_weights(file, metadata);
    
    

    return 0;
}

