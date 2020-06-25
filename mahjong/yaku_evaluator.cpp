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
        if (HasAllSimples(hand)) yaku.push_back(mj::Yaku::kAllSimples);

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

    bool YakuEvaluator::HasAllSimples(const Hand &hand) const noexcept {
        for (const Tile& tile : hand.ToVector()) {
            if (tile.Is(TileSetType::kYaocyu)) return false;
        }
        return true;
    }
}