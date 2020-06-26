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
        if (std::optional<int> score = HasAllSimples(hand); score) {
            yaku[Yaku::kAllSimples] = score.value();
        }
        if (std::optional<int> score = HasWhiteDragon(hand); score) {
            yaku[Yaku::kWhiteDragon] = score.value();
        }
        if (std::optional<int> score = HasGreenDragon(hand); score) {
            yaku[Yaku::kGreenDragon] = score.value();
        }
        if (std::optional<int> score = HasRedDragon(hand); score) {
            yaku[Yaku::kRedDragon] = score.value();
        }
        if (std::optional<int> score = HasAllTermsAndHonours(hand); score) {
            yaku[Yaku::kAllTermsAndHonours] = score.value();
        }
        if (std::optional<int> score = HasHalfFlush(hand); score) {
            yaku[Yaku::kHalfFlush] = score.value();
        }
        if (std::optional<int> score = HasFullFlush(hand); score) {
            yaku[Yaku::kFullFlush] = score.value();
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
            if (TotalFan(best_yaku) < TotalFan(yaku_in_this_pattern)) {
                std::swap(best_yaku, yaku_in_this_pattern);
            }
        }

        for (auto& [key, val] : best_yaku) yaku[key] = val;

        return yaku;
    }

    TileTypeCount YakuEvaluator::ClosedHandTiles(const Hand& hand) noexcept {
        TileTypeCount count;
        for (const Tile& tile : hand.ToVectorClosed(true)) {
            ++count[tile.Type()];
        }
        return count;
    }

    int YakuEvaluator::TotalFan(const std::map<Yaku,int>& yaku) noexcept {
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

    std::optional<int> YakuEvaluator::HasAllSimples(const Hand &hand) const noexcept {
        for (const Tile& tile : hand.ToVector()) {
            if (tile.Is(TileSetType::kYaocyu)) return std::nullopt;
        }
        return 1;
    }

    std::optional<int> YakuEvaluator::HasWhiteDragon(const Hand &hand) const noexcept {
        int total = 0;
        for (const Tile& tile : hand.ToVector()) {
            if (tile.Type() == TileType::kWD) ++total;
        }
        if (total < 2) return std::nullopt;
        return 1;
    }

    std::optional<int> YakuEvaluator::HasGreenDragon(const Hand &hand) const noexcept {
        int total = 0;
        for (const Tile& tile : hand.ToVector()) {
            if (tile.Type() == TileType::kGD) ++total;
        }
        if (total < 2) return std::nullopt;
        return 1;
    }

    std::optional<int> YakuEvaluator::HasRedDragon(const Hand &hand) const noexcept {
        int total = 0;
        for (const Tile& tile : hand.ToVector()) {
            if (tile.Type() == TileType::kRD) ++total;
        }
        if (total < 2) return std::nullopt;
        return 1;
    }

    std::optional<int> YakuEvaluator::HasAllTermsAndHonours(const Hand &hand) const noexcept {
        for (const Tile &tile : hand.ToVector()) {
            if (tile.Is(TileSetType::kTanyao)) return std::nullopt;
        }
        return 2;
    }

    std::optional<int> YakuEvaluator::HasHalfFlush(const Hand &hand) const noexcept {
        std::map<TileSetType,bool> set_types;
        for (const Tile& tile : hand.ToVector()) {
            if (tile.Is(TileSetType::kHonours)) set_types[TileSetType::kHonours] = true;
            else set_types[tile.Color()] = true;
        }

        if (set_types.count(TileSetType::kHonours) == 0 or set_types.size() > 2) return std::nullopt;
        if (hand.IsMenzen()) return 3;
        return 2;
    }

    std::optional<int> YakuEvaluator::HasFullFlush(const Hand &hand) const noexcept {
        std::map<TileSetType,bool> set_types;
        for (const Tile& tile : hand.ToVector()) {
            if (tile.Is(TileSetType::kHonours)) set_types[TileSetType::kHonours] = true;
            else set_types[tile.Color()] = true;
        }

        if (set_types.count(TileSetType::kHonours) or set_types.size() > 1) return std::nullopt;
        if (hand.IsMenzen()) return 6;
        return 5;
    }
}