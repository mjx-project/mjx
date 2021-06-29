#ifndef MAHJONG_SHANTEN_CACHE_H
#define MAHJONG_SHANTEN_CACHE_H

#include <string>
#include <unordered_map>
#include <vector>

namespace mjx::internal {
class ShantenCache {
 public:
  ShantenCache(const ShantenCache&) = delete;
  ShantenCache& operator=(ShantenCache&) = delete;
  ShantenCache(ShantenCache&&) = delete;
  ShantenCache operator=(ShantenCache&&) = delete;

  [[nodiscard]] static const ShantenCache& instance();
  [[nodiscard]] int Require(const std::vector<uint8_t>& count, int sets,
                            int heads) const;

 private:
  ShantenCache();
  ~ShantenCache() = default;
  void LoadCache();
  std::vector<std::vector<int>> cache_;
};
}  // namespace mjx::internal

#endif  // MAHJONG_SHANTEN_CACHE_H
