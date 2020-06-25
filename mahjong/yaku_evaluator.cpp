#include "yaku_evaluator.h"

#include <cassert>
#include <vector>

#include "types.h"

namespace mj
{
    YakuEvaluator::YakuEvaluator() : win_cache_() {}

    bool YakuEvaluator::Has(const Hand& hand) const noexcept {
        // TODO: 国士無双を判定する.
        auto [abstruct_hand, _] = WinningHandCache::CreateAbstructHand(ClosedHandTiles(hand));
        return win_cache_.Has(abstruct_hand);
    }

    std::map<Yaku, int> YakuEvaluator::Eval(Hand hand) const noexcept {

        assert(Has(hand));

        std::map<Yaku,int> yaku;

        // 手牌の組み合わせ方によらない役
        if (int score = HasFullyConcealdHand(hand); score) {
            yaku[Yaku::kFullyConcealedHand] = score;
        }


        // 手牌の組み合わせ方に依存する役
        auto [abstruct_hand, tile_types] = WinningHandCache::CreateAbstructHand(ClosedHandTiles(hand));

        std::map<Yaku,int> best_yaku;

        for (const auto& pattern : win_cache_.Patterns(abstruct_hand)) {
            std::vector<TileTypeCount> counts;
            for (const std::vector<int>& block : pattern) {
                TileTypeCount count;
                for (const int tile_type_id : block) {
                    ++count[tile_types[tile_type_id]];
                }
                counts.push_back(count);
            }

            for (const Open* open : hand.Opens()) {
                TileTypeCount count;
                for (const Tile tile : open->Tiles()) {
                    ++count[tile.Type()];
                }
                counts.push_back(count);
            }

            std::map<Yaku,int> yaku_in_this_pattern;

            if (int score = HasPinfu(hand); score) {
                yaku[Yaku::kPinfu] = score;
            }

            if (YakuScore(best_yaku) < YakuScore(yaku_in_this_pattern)) {
                std::swap(best_yaku, yaku_in_this_pattern);
            }
        }

        for (auto& [y, s] : best_yaku) yaku[y] = s;

        return yaku;
    }

    TileTypeCount YakuEvaluator::ClosedHandTiles(const Hand& hand) noexcept {
        TileTypeCount count;
        for (const Tile& tile : hand.ToVectorClosed(true)) {
            ++count[tile.Type()];
        }
        return count;
    }

    int YakuEvaluator::YakuScore(const std::map<Yaku,int>& yaku) noexcept {
        int total = 0;
        for (const auto& [y, s] : yaku) total += s;
        return total;
    }

    bool YakuEvaluator::HasFullyConcealdHand(const Hand &hand) const noexcept {
        return hand.IsMenzen() and hand.Stage() == HandStage::kAfterTsumo;
    }

    bool YakuEvaluator::HasPinfu(const Hand &hand) const noexcept {
        return true;
    }
}