#ifndef MAHJONG_WIN_CACHE_H
#define MAHJONG_WIN_CACHE_H

#include "types.h"

#include <memory>
#include <map>
#include <unordered_map>
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
        std::pair<AbstructHand, std::map<int, TileType>>
        CreateAbstructHand(const std::map<TileType, int>& count) const noexcept ;
    private:
        std::map<AbstructHand, std::vector<SplitPattern>> cache_;
        //std::map<std::string, std::vector<int>> cache_;
        std::vector<std::map<TileType, int>> CreateSets() const noexcept ;
        std::vector<std::map<TileType, int>> CreateHeads() const noexcept ;
    };
}

#endif //MAHJONG_WIN_CACHE_H
