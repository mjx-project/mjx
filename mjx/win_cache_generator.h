#ifndef MAHJONG_WIN_CACHE_GENERATOR_H
#define MAHJONG_WIN_CACHE_GENERATOR_H

#include <memory>
#include <map>
#include <unordered_map>
#include <set>
#include <vector>

#include "win_cache.h"
#include "types.h"

namespace mjx
{
    class WinHandCacheGenerator
    {
    public:
        static void GenerateCache() noexcept ;
        static void GenerateTenpaiCache() noexcept ;
    private:
        [[nodiscard]] static std::vector<TileTypeCount> CreateSets() noexcept ;
        [[nodiscard]] static std::vector<TileTypeCount> CreateHeads() noexcept ;
        static bool Register(
                const std::vector<TileTypeCount>& blocks,
                const TileTypeCount& total,
                WinHandCache::CacheType& cache) noexcept ;
        static void Add(TileTypeCount& total, const TileTypeCount& block) noexcept ;
        static void Sub(TileTypeCount& total, const TileTypeCount& block) noexcept ;
        static std::unordered_set<AbstructHand> ReduceTile(const AbstructHand& hand) noexcept ;
        static void ShowStatus(const WinHandCache::CacheType& cache) noexcept ;
    };
}  // namespace mjx

#endif //MAHJONG_WIN_CACHE_GENERATOR_H
