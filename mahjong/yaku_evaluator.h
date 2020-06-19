#ifndef MAHJONG_YAKU_EVALUATOR_H
#define MAHJONG_YAKU_EVALUATOR_H

#include <vector>

#include "types.h"
#include "hand.h"
#include "win_cache.h"

namespace mj
{
    class YakuEvaluator {
    public:
        YakuEvaluator();
        bool Has(const Hand& hand) const noexcept ;
        std::vector<Yaku> Eval(const Hand& hand) const noexcept ;
    private:

        [[nodiscard]] static std::pair<AbstructHand, std::vector<TileType>>
        CreateAbstructHand(const Hand& hand) noexcept ;
        bool HasFullyConcealdHand(const Hand& hand) const noexcept ;
        WinningHandCache win_cache_;
    };
} // namespace mj

#endif //MAHJONG_YAKU_EVALUATOR_H
