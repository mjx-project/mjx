#ifndef MAHJONG_WIN_CACHE_H
#define MAHJONG_WIN_CACHE_H

#include <unordered_map>
#include <set>
#include <map>
#include <vector>
#include <string>

#include "types.h"

namespace mj
{
    class WinningHandCache
    {
    public:
        WinningHandCache();
        [[nodiscard]] static std::pair<win_cache::AbstructHand, std::vector<TileType>>
        CreateAbstructHand(const TileTypeCount& count) noexcept ;
        [[nodiscard]] bool Has(const TileTypeCount& closed_hand) const noexcept ;
        [[nodiscard]] bool Has(const std::string& abstruct_hand) const noexcept ;
        [[nodiscard]] std::vector<std::pair<std::vector<TileTypeCount>, std::vector<TileTypeCount>>>
        SetAndHeads(const TileTypeCount& closed_hand) const noexcept ;
    private:
        win_cache::CacheType cache_;
        void LoadWinCache();
    };
}  // namespace mj

#endif //MAHJONG_WIN_CACHE_H
