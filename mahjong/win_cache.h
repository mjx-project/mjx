#ifndef MAHJONG_CLOSED_WIN_CACHE_H
#define MAHJONG_CLOSED_WIN_CACHE_H

#include <memory>
#include <unordered_map>

namespace mj
{
    using AbstructHand = std::vector<std::vector<int>>;

    struct AbstructHandInfo {
        std::pair<int,int> head_pos;
        std::vector<std::pair<int,int>> triple_pos;
    };

    class WinningClosedHandCache
    {
    public:
        WinningClosedHandCache();
        [[nodiscard]] bool Has(const std::vector<Tile> &closed_hand) const noexcept ;
        // utils
        static void PrepareWinCache();
    private:
        std::unique_ptr<std::unordered_map<AbstructHand, AbstructHandInfo>> cache_;

        using TileCount = std::array<int,34>;
        std::vector<TileCount> CreateSets() const noexcept ;
        std::vector<TileCount> CreateHeads() const noexcept ;
    };
}

#endif //MAHJONG_CLOSED_WIN_CACHE_H
