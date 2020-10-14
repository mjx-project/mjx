#include <iostream>
#include <dataset/train_data_generator.h>

int main() {
    std::cerr << "Hello" << std::endl;
    mj::TrainDataGenerator::generate(
            std::string(RESOURCES_DIR) + "/2010091009gm-00a9-0000-83af2648&tw=2.json",
            std::string(RESOURCES_DIR) + "/train_data.txt");
    return 0;
}
