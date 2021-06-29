#include "shanten_cache.h"

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <iostream>
#include <sstream>
#include <numeric>

namespace mjx::internal {
ShantenCache::ShantenCache() { LoadCache(); }

const ShantenCache& ShantenCache::instance() {
  static ShantenCache
      instance;  // Thread safe from C++ 11
                 // https://cpprefjp.github.io/lang/cpp11/static_initialization_thread_safely.html
  return instance;
}
int ShantenCache::Require(const std::vector<uint8_t>& count, int sets,
                          int heads) const {
  assert(count.size() == 9);
  int code = 0;
  for (int i = 0; i < 9; ++i) {
    code = code * 5 + count[i];
  }
  return cache_[code][heads * 5 + sets];
}

void ShantenCache::LoadCache() {
  std::cerr << "ShantenCache::LoadCache: start" << std::endl;

  std::ifstream ifs(std::string(WIN_CACHE_DIR) + "/shanten-rs.txt", std::ios::in);
  std::string line;
  while (!ifs.eof()) {
    std::getline(ifs, line);
    if (line.empty()) break;
    std::stringstream ss(line);
    std::vector<int> c(10);
    for (int i = 0; i < 10; ++i) ss >> c[i];
    cache_.push_back(c);
  }
  assert(cache_.size() == 1953125);
  std::cerr << "ShantenCache::LoadCache: end" << std::endl;
}
}  // namespace mjx::internal
