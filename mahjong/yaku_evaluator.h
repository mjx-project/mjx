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
        [[nodiscard]] bool Has(const Hand& hand) const noexcept ;
        [[nodiscard]] std::map<Yaku,int> Eval(Hand hand) const noexcept ;

    private:
        static TileTypeCount ClosedHandTiles(const Hand& hand) noexcept ;
        static int YakuScore(const std::map<Yaku,int>& yaku) noexcept ;
        [[nodiscard]] bool HasFullyConcealdHand(const Hand& hand) const noexcept ;
        [[nodiscard]] bool HasPinfu(const Hand& hand) const noexcept ;
        WinningHandCache win_cache_;
    };
} // namespace mj

#endif //MAHJONG_YAKU_EVALUATOR_H
