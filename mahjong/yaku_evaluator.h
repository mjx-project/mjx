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
        [[nodiscard]] std::map<Yaku,int> Eval(const Hand& hand) const noexcept ;

    private:
        static TileTypeCount ClosedHandTiles(const Hand& hand) noexcept ;
        static int TotalFan(const std::map<Yaku,int>& yaku) noexcept ;
        [[nodiscard]] std::optional<int> HasAllSimples(const Hand& hand) const noexcept ;
        [[nodiscard]] std::optional<int> HasWhiteDragon(const Hand& hand) const noexcept ;
        [[nodiscard]] std::optional<int> HasGreenDragon(const Hand& hand) const noexcept ;
        [[nodiscard]] std::optional<int> HasRedDragon(const Hand& hand) const noexcept ;
        [[nodiscard]] std::optional<int> HasAllTermsAndHonours(const Hand& hand) const noexcept ;
        [[nodiscard]] std::optional<int> HasHalfFlush(const Hand &hand) const noexcept ;
        [[nodiscard]] std::optional<int> HasFullFlush(const Hand &hand) const noexcept ;
        [[nodiscard]] std::optional<int> HasFullyConcealdHand(const Hand& hand) const noexcept ;
        [[nodiscard]] std::optional<int> HasPinfu(
                const Hand &hand,
                const std::vector<TileTypeCount>& sets,
                const std::vector<TileTypeCount>& heads) const noexcept ;
        [[nodiscard]] std::optional<int> HasPureDoubleChis(
                const Hand &hand,
                const std::vector<TileTypeCount>& sets,
                const std::vector<TileTypeCount>& heads) const noexcept ;
        [[nodiscard]] std::optional<int> HasTwicePureDoubleChis(
                const Hand &hand,
                const std::vector<TileTypeCount>& sets,
                const std::vector<TileTypeCount>& heads) const noexcept ;
        [[nodiscard]] std::optional<int> HasSevenPairs(
                const Hand &hand,
                const std::vector<TileTypeCount>& sets,
                const std::vector<TileTypeCount>& heads) const noexcept ;
        WinningHandCache win_cache_;
    };
} // namespace mj

#endif //MAHJONG_YAKU_EVALUATOR_H
