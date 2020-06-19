#include "yaku_evaluator.h"

#include <cassert>
#include <vector>

namespace mj
{
    YakuEvaluator::YakuEvaluator() : win_cache_() {}

    bool YakuEvaluator::Has(const Hand& hand) const noexcept {
        auto [abstruct_hand, _] = CreateAbstructHand(hand);
        return win_cache_.Has(abstruct_hand);
    }

    std::vector<Yaku> YakuEvaluator::Eval(const Hand& hand) const noexcept {

        assert(Has(hand));

        std::vector<Yaku> yaku;

        if (HasFullyConcealdHand(hand)) yaku.push_back(mj::Yaku::kFullyConcealedHand);

        return yaku;
    }

    std::pair<AbstructHand, std::vector<TileType>>
    YakuEvaluator::CreateAbstructHand(const Hand& hand) noexcept {
        mj::TileCount count;
        for (const Tile& tile : hand.ToVectorClosed(true)) {
            ++count[tile.Type()];
        }
        return WinningHandCache::CreateAbstructHand(count);
    }

    bool YakuEvaluator::HasFullyConcealdHand(const Hand &hand) const noexcept {
        return hand.IsMenzen() and hand.Stage() == mj::HandStage::kAfterTsumo;
    }
}