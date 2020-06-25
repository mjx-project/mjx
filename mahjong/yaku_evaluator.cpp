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

    std::map<Yaku, int> YakuEvaluator::Eval(const Hand& hand) const noexcept {

        assert(Has(hand));

        std::map<Yaku,int> yaku;

        // 手牌の組み合わせ方によらない役
        if (std::optional<int> score = HasFullyConcealdHand(hand); score) {
            yaku[Yaku::kFullyConcealedHand] = score.value();
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

            // 各役の判定
            if (std::optional<int> score = HasPinfu(hand, counts); score) {
                yaku[Yaku::kPinfu] = score.value();
            }

            // 今までに調べた組み合わせ方より役の総得点が高いなら採用する.
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

    std::optional<int> YakuEvaluator::HasFullyConcealdHand(const Hand &hand) const noexcept {
        if (hand.IsMenzen() and hand.Stage() == HandStage::kAfterTsumo) return 1;
        return std::nullopt;
    }

    std::optional<int> YakuEvaluator::HasPinfu(const Hand &hand, const std::vector<TileTypeCount>& counts) const noexcept {
        if (!hand.IsMenzen()) return std::nullopt;

        std::vector<TileTypeCount> sets, heads;

        for (const TileTypeCount& count : counts) {
            switch (count.size()) {
                case 3:
                    sets.push_back(count);
                    break;
                case 1:
                    if (count.begin()->second == 2) {
                        heads.push_back(count);
                    } else {
                        return std::nullopt;
                    }
                    break;
                default:
                    return std::nullopt;
            }
        }

        if (sets.size() != 4 or heads.size() != 1) return std::nullopt;

        // TODO: 場風, 自風も弾く.
        if (const TileType head = heads[0].begin()->first;
                        head == TileType::kRD or
                        head == TileType::kGD or
                        head == TileType::kWD) {
            return std::nullopt;
        }

        assert(hand.LastTileAdded().has_value());
        const Tile tsumo = hand.LastTileAdded().value();

        for (const TileTypeCount& st : sets) {
            const TileType left = st.begin()->first,
                           right = st.rbegin()->first;
            if ((tsumo.Type() == left and Num(right) != 9) or
                (tsumo.Type() == right and Num(left) != 1)) {

                // いずれかの順子のリャンメン待ちをツモっていたらOK
                return 1;
            }
        }

        return std::nullopt;
    }
}