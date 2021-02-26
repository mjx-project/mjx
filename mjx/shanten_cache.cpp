#include "shanten_cache.h"

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <numeric>
#include <iostream>

namespace mjx {
    ShantenCache::ShantenCache() {
        LoadCache();
    }

    const ShantenCache& ShantenCache::instance() {
        static ShantenCache instance;  // Thread safe from C++ 11  https://cpprefjp.github.io/lang/cpp11/static_initialization_thread_safely.html
        return instance;
    }
    int ShantenCache::Require(const std::vector<uint8_t>& count, int sets, int heads) const {
        if (sets == 0 and heads == 0) return 0;

        assert(count.size() == 9);
        assert(std::accumulate(count.begin(), count.end(), 0) <= 14);
        std::string code;
        for (int i = 0; i < 9; ++i) {
            code += std::to_string(count[i]);
        }
        code += '-';
        code += std::to_string(sets);
        code += '-';
        code += std::to_string(heads);

        return cache_.at(code);
    }

    void ShantenCache::LoadCache() {
        std::cerr << "ShantenCache::LoadCache: start" << std::endl;
        boost::property_tree::ptree root;
        boost::property_tree::read_json(std::string(WIN_CACHE_DIR) + "/shanten_cache.json", root);
        cache_.reserve(root.size());
        int mx = 0;
        for (const auto& [hand, patterns_pt] : root) {
            cache_[hand] = patterns_pt.get_value<int>();
            mx = std::max(mx, cache_[hand]);
        }
        assert(mx > 0);
        std::cerr << "Max:" << mx << std::endl;
        std::cerr << "ShantenCache::LoadCache: end" << std::endl;
    }
}
