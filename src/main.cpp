#include "model/Model.hpp"
#include <iostream>
#include <fstream>
#include "model/WeightLoader.hpp"
#include "model/Config.hpp"

using namespace std;

int main() {
    
    WeightLoader loader;

    string filename =""

    istream file(filename);
    loader.load(filename);
    
    MetadataMap = loader.get_tensor_map();

    Config config;
    Model model(config);

    model.load_weights(file, metadata);

    

    return 0;
}
