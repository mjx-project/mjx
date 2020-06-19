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
        [[nodiscard]] bool Has(const std::string &abstruct_hand) const noexcept ;
        [[nodiscard]] static std::pair<win_cache::AbstructHand, std::vector<TileType>>
        CreateAbstructHand(const TileTypeCount& count) noexcept ;
        [[nodiscard]] const std::set<win_cache::SplitPattern>& Patterns(
                const win_cache::AbstructHand &abstruct_hand) const noexcept ;
    private:
        win_cache::CacheType cache_;
        void LoadWinCache();
    };
}  // namespace mj

#endif //MAHJONG_WIN_CACHE_H
