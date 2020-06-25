#include "yaku_evaluator.h"

#include <cassert>
#include <vector>

#include "types.h"

namespace mj
{
    YakuEvaluator::YakuEvaluator() : win_cache_() {}

    bool YakuEvaluator::Has(const Hand& hand) const noexcept {
        auto [abstruct_hand, _] = WinningHandCache::CreateAbstructHand(ClosedHandTiles(hand));
        return win_cache_.Has(abstruct_hand);
    }

    std::vector<Yaku> YakuEvaluator::Eval(const Hand& hand) const noexcept {

        assert(Has(hand));

        std::vector<Yaku> yaku;

        if (HasFullyConcealdHand(hand)) yaku.push_back(mj::Yaku::kFullyConcealedHand);
        if (HasWhiteDragon(hand)) yaku.push_back(mj::Yaku::kWhiteDragon);
        if (HasGreenDragon(hand)) yaku.push_back(mj::Yaku::kGreenDragon);
        if (HasRedDragon(hand)) yaku.push_back(mj::Yaku::kRedDragon);

        return yaku;
    }

    TileTypeCount YakuEvaluator::ClosedHandTiles(const Hand& hand) noexcept {
        TileTypeCount count;
        for (const Tile& tile : hand.ToVectorClosed(true)) {
            ++count[tile.Type()];
        }
        return count;
    }

    bool YakuEvaluator::HasFullyConcealdHand(const Hand &hand) const noexcept {
        return hand.IsMenzen() and hand.Stage() == HandStage::kAfterTsumo;
    }

    bool YakuEvaluator::HasWhiteDragon(const Hand &hand) const noexcept {
        int total = 0;
        for (const Tile& tile : hand.ToVector()) {
            if (tile.Type() == TileType::kWD) ++total;
        }
        return total >= 3;
    }
    bool YakuEvaluator::HasGreenDragon(const Hand &hand) const noexcept {
        int total = 0;
        for (const Tile& tile : hand.ToVector()) {
            if (tile.Type() == TileType::kGD) ++total;
        }
        return total >= 3;
    }
    bool YakuEvaluator::HasRedDragon(const Hand &hand) const noexcept {
        int total = 0;
        for (const Tile& tile : hand.ToVector()) {
            if (tile.Type() == TileType::kRD) ++total;
        }
        return total >= 3;
    }
}