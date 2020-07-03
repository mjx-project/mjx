#ifndef MAHJONG_WIN_CACHE_GENERATOR_H
#define MAHJONG_WIN_CACHE_GENERATOR_H

#include <memory>
#include <map>
#include <unordered_map>
#include <set>
#include <vector>

#include "win_cache.h"
#include "types.h"

namespace mj
{
    class WinningHandCacheGenerator
    {
    public:
        static void GenerateCache() noexcept ;
    private:
        [[nodiscard]] static std::vector<TileTypeCount> CreateSets() noexcept ;
        [[nodiscard]] static std::vector<TileTypeCount> CreateHeads() noexcept ;
        static bool Register(
                const std::vector<TileTypeCount>& blocks,
                const TileTypeCount& total,
                WinningHandCache::CacheType& cache) noexcept ;
        static void Add(TileTypeCount& total, const TileTypeCount& block) noexcept ;
        static void Sub(TileTypeCount& total, const TileTypeCount& block) noexcept ;
        static void ShowStatus(const WinningHandCache::CacheType& cache) noexcept ;
    };
}  // namespace mj

#endif //MAHJONG_WIN_CACHE_GENERATOR_H
