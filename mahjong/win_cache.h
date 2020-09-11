#ifndef MAHJONG_WIN_CACHE_H
#define MAHJONG_WIN_CACHE_H

#include <unordered_map>
#include <set>
#include <map>
#include <vector>
#include <string>

#include "types.h"
#include "abstruct_hand.h"

namespace mj
{
    class WinHandCache
    {
    public:
        WinHandCache(const WinHandCache&) = delete;
        WinHandCache& operator=(WinHandCache&) = delete;
        WinHandCache(WinHandCache&&) = delete;
        WinHandCache operator=(WinHandCache&&) = delete;

        static const WinHandCache& instance();

        [[nodiscard]] bool Has(const TileTypeCount& closed_hand) const noexcept ;
        [[nodiscard]] std::unordered_set<TileType> Machi(TileTypeCount closed_hand) const noexcept ;
        [[nodiscard]] std::vector<std::pair<std::vector<TileTypeCount>, std::vector<TileTypeCount>>>
        SetAndHeads(const TileTypeCount& closed_hand) const noexcept ;

        using SplitPattern = std::vector<std::vector<int>>;
        using CacheType = std::unordered_map<AbstructHand, std::set<SplitPattern>>;
    private:
        WinHandCache();
        ~WinHandCache() = default;
        CacheType cache_;
        void LoadWinCache();
    };
}  // namespace mj

#endif //MAHJONG_WIN_CACHE_H
