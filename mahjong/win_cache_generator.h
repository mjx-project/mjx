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
        [[nodiscard]] static std::vector<TileCount> CreateSets() noexcept ;
        [[nodiscard]] static std::vector<TileCount> CreateHeads() noexcept ;
        static bool Register(const std::vector<TileCount>& blocks, const TileCount& total, CacheType& cache) noexcept ;
        static void Add(TileCount& total, const TileCount& block) noexcept ;
        static void Sub(TileCount& total, const TileCount& block) noexcept ;
        static void ShowStatus(const CacheType& cache) noexcept ;
    };
}  // namespace mj

#endif //MAHJONG_WIN_CACHE_GENERATOR_H
