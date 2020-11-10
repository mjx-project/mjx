#include <string>
#include "mj/observation.h"
#include "mj/state.h"

namespace mj {
    class TrainDataGenerator {
    public:
        static void generateDiscard(const std::string& src_path, const std::string& dst_path);
        static void generateOpen(const std::string& src_path, const std::string& dst_path, mjproto::ActionType open_type);
        static void generateOpenYesNo(const std::string& src_path, const std::string& dst_path, mjproto::ActionType open_type);
    };
}