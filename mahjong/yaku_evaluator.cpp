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
        if (HasHalfFlush(hand)) yaku.push_back(mj::Yaku::kHalfFlush);
        if (HasFullFlush(hand)) yaku.push_back(mj::Yaku::kFullFlush);

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

    bool YakuEvaluator::HasHalfFlush(const Hand &hand) const noexcept {
        std::map<TileSetType,bool> set_types;
        for (const Tile& tile : hand.ToVector()) {
            if (tile.Is(TileSetType::kHonours)) set_types[TileSetType::kHonours] = true;
            else set_types[tile.Color()] = true;
        }

        return set_types.count(TileSetType::kHonours) and set_types.size() == 2;
    }

    bool YakuEvaluator::HasFullFlush(const Hand &hand) const noexcept {
        std::map<TileSetType,bool> set_types;
        for (const Tile& tile : hand.ToVector()) {
            if (tile.Is(TileSetType::kHonours)) set_types[TileSetType::kHonours] = true;
            else set_types[tile.Color()] = true;
        }

        return set_types.count(TileSetType::kHonours) == 0 and set_types.size() == 1;
    }
}