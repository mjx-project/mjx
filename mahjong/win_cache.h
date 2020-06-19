#ifndef MAHJONG_WIN_CACHE_H
#define MAHJONG_WIN_CACHE_H

#include <unordered_map>
#include <set>
#include <vector>
#include <string>

#include "types.h"

namespace mj
{
    using AbstructHand = std::string;
    using SplitPattern = std::vector<std::vector<int>>;
    using CacheType = std::unordered_map<AbstructHand, std::set<SplitPattern>>;

    class WinningHandCache
    {
    public:
        WinningHandCache();
        [[nodiscard]] bool Has(const std::string &s) const noexcept ;
    private:
        CacheType cache_;
        void LoadWinCache();
    };
}  // namespace mj

#endif //MAHJONG_WIN_CACHE_H
