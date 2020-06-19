#ifndef MAHJONG_WIN_CACHE_H
#define MAHJONG_WIN_CACHE_H

#include "types.h"

#include <memory>
#include <map>
#include <unordered_map>
#include <set>
#include <vector>
#include <string>

namespace mj
{
    using AbstructHand = std::string;
    using SplitPattern = std::vector<std::vector<int>>;

    class WinningHandCache
    {
    public:
        WinningHandCache();
        [[nodiscard]] bool Has(const std::string &s) const noexcept ;
    private:
        std::map<AbstructHand, std::set<SplitPattern>> cache_;
        void LoadWinCache();
    };
}

#endif //MAHJONG_WIN_CACHE_H
