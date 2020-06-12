#ifndef MAHJONG_WIN_CACHE_H
#define MAHJONG_WIN_CACHE_H

#include "types.h"

#include <memory>
#include <map>
#include <unordered_map>
#include <set>
#include <vector>

namespace mj
{
    using AbstructHand = std::string;
    using SplitPattern = std::vector<std::vector<int>>;

    class WinningHandCache
    {
    public:
        WinningHandCache();
        [[nodiscard]] bool Has(const std::string &s) const noexcept ;
        void PrepareWinCache();
        void LoadWinCache();
        [[nodiscard]] std::pair<AbstructHand, std::map<int, TileType>>
        CreateAbstructHand(const std::map<TileType, int>& count) const noexcept ;
    private:
        std::unordered_map<AbstructHand, std::set<SplitPattern>> cache_;
        [[nodiscard]] std::vector<std::map<TileType, int>> CreateSets() const noexcept ;
        [[nodiscard]] std::vector<std::map<TileType, int>> CreateHeads() const noexcept ;
    };
}

#endif //MAHJONG_WIN_CACHE_H
