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
        [[nodiscard]] std::pair<AbstructHand, std::vector<TileType>>
        CreateAbstructHand(const std::map<TileType, int>& count) const noexcept ;
    private:
        std::map<AbstructHand, std::set<SplitPattern>> cache_;
        void PrepareWinCache();
        void LoadWinCache();
        [[nodiscard]] std::vector<std::map<TileType, int>> CreateSets() const noexcept ;
        [[nodiscard]] std::vector<std::map<TileType, int>> CreateHeads() const noexcept ;
        bool Register(const std::vector<std::map<TileType,int>>& blocks, const std::map<TileType,int>& total);
        void Add(std::map<TileType,int>& total, const std::map<TileType,int>& block);
        void Sub(std::map<TileType,int>& total, const std::map<TileType,int>& block);
        void ShowStatus() const noexcept;
    };
}

#endif //MAHJONG_WIN_CACHE_H
