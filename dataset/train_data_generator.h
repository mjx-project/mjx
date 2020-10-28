#include <string>
#include "mj/observation.h"
#include "mj/state.h"

namespace mj {
    class TrainDataGenerator {
    public:
        static void generateDiscard(const std::string& src_path, const std::string& dst_path);
        static void generateChi(const std::string& src_path, const std::string& dst_path);
    };
}