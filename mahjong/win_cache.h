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
    class WinningHandCache
    {
    public:
        WinningHandCache(const WinningHandCache&) = delete;
        WinningHandCache& operator=(WinningHandCache&) = delete;
        WinningHandCache(WinningHandCache&&) = delete;
        WinningHandCache operator=(WinningHandCache&&) = delete;

        static const WinningHandCache& get_instance();

        [[nodiscard]] bool Has(const TileTypeCount& closed_hand) const noexcept ;
        [[nodiscard]] bool Has(const std::string& abstruct_hand) const noexcept ;
        [[nodiscard]] std::vector<std::pair<std::vector<TileTypeCount>, std::vector<TileTypeCount>>>
        SetAndHeads(const TileTypeCount& closed_hand) const noexcept ;

        using SplitPattern = std::vector<std::vector<int>>;
        using CacheType = std::unordered_map<AbstructHand, std::set<SplitPattern>>;
    private:
        WinningHandCache();
        ~WinningHandCache() = default;
        CacheType cache_;
        void LoadWinCache();
    };
}  // namespace mj

#endif //MAHJONG_WIN_CACHE_H
