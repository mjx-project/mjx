#ifndef MAHJONG_WIN_CACHE_GENERATOR_H
#define MAHJONG_WIN_CACHE_GENERATOR_H

#include <map>
#include <memory>
#include <set>
#include <unordered_map>
#include <vector>

#include "mjx/internal/types.h"
#include "mjx/internal/win_cache.h"

namespace mjx::internal {
class WinHandCacheGenerator {
 public:
  static void GenerateCache() noexcept;
  static void GenerateTenpaiCache() noexcept;

 private:
  [[nodiscard]] static std::vector<TileTypeCount> CreateSets() noexcept;
  [[nodiscard]] static std::vector<TileTypeCount> CreateHeads() noexcept;
  static bool Register(const std::vector<TileTypeCount>& blocks,
                       const TileTypeCount& total,
                       WinHandCache::CacheType& cache) noexcept;
  static void Add(TileTypeCount& total, const TileTypeCount& block) noexcept;
  static void Sub(TileTypeCount& total, const TileTypeCount& block) noexcept;
  static std::unordered_set<AbstructHand> ReduceTile(
      const AbstructHand& hand) noexcept;
  static void ShowStatus(const WinHandCache::CacheType& cache) noexcept;
};
}  // namespace mjx::internal

#endif  // MAHJONG_WIN_CACHE_GENERATOR_H
