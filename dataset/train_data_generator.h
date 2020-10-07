#include "mj/mj.h"
#include <string>

namespace mj {
    class TrainDataGenerator {
    public:
        void generate(const std::string& src_path, const std::string& dst_path);
    };
}