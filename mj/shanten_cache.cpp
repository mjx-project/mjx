#include "shanten_cache.h"

namespace mj {
    ShantenCache::ShantenCache() {
        // LoadCache();
    }

    const ShantenCache& ShantenCache::instance() {
        static ShantenCache instance;  // Thread safe from C++ 11  https://cpprefjp.github.io/lang/cpp11/static_initialization_thread_safely.html
        return instance;
    }
    int ShantenCache::Require(const std::vector<uint8_t>& count, int sets, int heads) const {
        // TODO: implement
        return 0;
    }
}
