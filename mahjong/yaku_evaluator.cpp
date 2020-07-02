#include "yaku_evaluator.h"

#include <cassert>
#include <vector>

#include "types.h"

namespace mj
{
    YakuEvaluator::YakuEvaluator() : win_cache_() {}

    bool YakuEvaluator::Has(const Hand& hand) const noexcept {
        // TODO: 国士無双を判定する.
        return win_cache_.Has(ClosedHandTiles(hand));
    }

    std::map<Yaku, int> YakuEvaluator::Eval(const Hand& hand) const noexcept {

        assert(Has(hand));

        std::map<Yaku,int> yaku;

        // 手牌の組み合わせ方によらない役
        TileTypeCount all_tiles = ClosedAndOpenedHandTiles(hand);
        if (std::optional<int> score = HasFullyConcealdHand(hand); score) {
            yaku[Yaku::kFullyConcealedHand] = score.value();
        }
        if (std::optional<int> score = HasAllSimples(all_tiles); score) {
            yaku[Yaku::kAllSimples] = score.value();
        }
        if (std::optional<int> score = HasWhiteDragon(all_tiles); score) {
            yaku[Yaku::kWhiteDragon] = score.value();
        }
        if (std::optional<int> score = HasGreenDragon(all_tiles); score) {
            yaku[Yaku::kGreenDragon] = score.value();
        }
        if (std::optional<int> score = HasRedDragon(all_tiles); score) {
            yaku[Yaku::kRedDragon] = score.value();
        }
        if (std::optional<int> score = HasAllTermsAndHonours(all_tiles); score) {
            yaku[Yaku::kAllTermsAndHonours] = score.value();
        }
        if (std::optional<int> score = HasHalfFlush(hand, all_tiles); score) {
            yaku[Yaku::kHalfFlush] = score.value();
        }
        if (std::optional<int> score = HasFullFlush(hand, all_tiles); score) {
            yaku[Yaku::kFullFlush] = score.value();
        }

        // 手牌の組み合わせ方に依存する役
        std::map<Yaku,int> best_yaku;

        std::vector<TileTypeCount> opened_sets;
        for (const Open* open : hand.Opens()) {
            TileTypeCount count;
            for (const Tile tile : open->Tiles()) {
                ++count[tile.Type()];
            }
            opened_sets.push_back(count);
        }

        for (auto [closed_sets, heads] : win_cache_.SetAndHeads(ClosedHandTiles(hand))) {

            std::map<Yaku,int> yaku_in_this_pattern;

            // 各役の判定
            if (std::optional<int> score = HasPinfu(hand, closed_sets, opened_sets, heads); score) {
                yaku_in_this_pattern[Yaku::kPinfu] = score.value();
            }
            if (std::optional<int> score = HasPureDoubleChis(hand, closed_sets, opened_sets, heads); score) {
                yaku_in_this_pattern[Yaku::kPureDoubleChis] = score.value();
            }
            if (std::optional<int> score = HasTwicePureDoubleChis(hand, closed_sets, opened_sets, heads); score) {
                yaku_in_this_pattern[Yaku::kTwicePureDoubleChis] = score.value();
            }
            if (std::optional<int> score = HasSevenPairs(hand, closed_sets, opened_sets, heads); score) {
                yaku_in_this_pattern[Yaku::kSevenPairs] = score.value();
            }
            if (std::optional<int> score = HasAllPons(hand, closed_sets, opened_sets, heads); score) {
                yaku_in_this_pattern[Yaku::kAllPons] = score.value();
            }
            if (std::optional<int> score = HasPureStraight(hand, closed_sets, opened_sets, heads); score) {
                yaku_in_this_pattern[Yaku::kPureStraight] = score.value();
            }
            if (std::optional<int> score = HasMixedTripleChis(hand, closed_sets, opened_sets, heads); score) {
                yaku_in_this_pattern[Yaku::kMixedTripleChis] = score.value();
            }
            if (std::optional<int> score = HasTriplePons(hand, closed_sets, opened_sets, heads); score) {
                yaku_in_this_pattern[Yaku::kTriplePons] = score.value();
            }
            if (std::optional<int> score = HasOutsideHand(hand, closed_sets, opened_sets, heads); score) {
                yaku_in_this_pattern[Yaku::kOutsideHand] = score.value();
            }
            if (std::optional<int> score = HasTerminalsInAllSets(hand, closed_sets, opened_sets, heads); score) {
                yaku_in_this_pattern[Yaku::kTerminalsInAllSets] = score.value();
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
    TileTypeCount YakuEvaluator::ClosedAndOpenedHandTiles(const Hand& hand) noexcept {
        TileTypeCount count;
        for (const Tile& tile : hand.ToVector(true)) {
            ++count[tile.Type()];
        }
        return count;
    }

    int YakuEvaluator::TotalFan(const std::map<Yaku,int>& yaku) noexcept {
        int total = 0;
        for (const auto& [key, val] : yaku) total += val;
        return total;
    }

    std::optional<int> YakuEvaluator::HasPinfu(
            const Hand &hand,
            const std::vector<TileTypeCount>& closed_sets,
            const std::vector<TileTypeCount>& opened_sets,
            const std::vector<TileTypeCount>& heads) const noexcept {
        if (!hand.IsMenzen()) return std::nullopt;

        if (closed_sets.size() != 4 or heads.size() != 1) {
            // 基本形でなければNG.
            return std::nullopt;
        }

        for (const TileTypeCount& count : closed_sets) {
            if (count.size() == 1) {
                // 刻子が含まれるとNG
                return std::nullopt;
            }
        }

        // 雀頭が役牌だとNG
        // TODO: 場風, 自風も弾く.
        if (const TileType head = heads[0].begin()->first;
                        head == TileType::kRD or
                        head == TileType::kGD or
                        head == TileType::kWD) {
            return std::nullopt;
        }

        assert(hand.LastTileAdded().has_value());
        const Tile tsumo = hand.LastTileAdded().value();

        for (const TileTypeCount& st : closed_sets) {
            const TileType left = st.begin()->first,
                           right = st.rbegin()->first;
            if ((tsumo.Type() == left and Num(right) != 9) or
                (tsumo.Type() == right and Num(left) != 1)) {

                // いずれかの順子のリャンメン待ちをツモっていたらOK
                return 1;
            }
        }

        // リャンメン待ちでなければNG
        return std::nullopt;
    }

    std::optional<int> YakuEvaluator::HasPureDoubleChis(
            const Hand &hand,
            const std::vector<TileTypeCount>& closed_sets,
            const std::vector<TileTypeCount>& opened_sets,
            const std::vector<TileTypeCount>& heads) const noexcept {

        // 鳴いているとNG
        if (!hand.IsMenzen()) return std::nullopt;

        if (closed_sets.size() != 4 or heads.size() != 1) {
            // 基本形でなければNG.
            return std::nullopt;
        }

        std::map<TileTypeCount, int> pure_chis;
        for (const TileTypeCount& count : closed_sets) {
            if (count.size() == 3) ++pure_chis[count];
        }

        int pairs = 0;
        for (auto& [pure_chie, n] : pure_chis) {
            if (n >= 2) ++pairs;
        }

        // 2つ以上重なっている順子が1個だけあるならOK
        if (pairs == 1) return 1;

        return std::nullopt;
    }

    std::optional<int> YakuEvaluator::HasTwicePureDoubleChis(
            const Hand &hand,
            const std::vector<TileTypeCount>& closed_sets,
            const std::vector<TileTypeCount>& opened_sets,
            const std::vector<TileTypeCount>& heads) const noexcept {

        // 鳴いているとNG
        if (!hand.IsMenzen()) return std::nullopt;

        if (closed_sets.size() != 4 or heads.size() != 1) {
            // 基本形でなければNG.
            return std::nullopt;
        }

        std::map<TileTypeCount, int> pure_chis;
        for (const TileTypeCount& count : closed_sets) {
            if (count.size() == 3) ++pure_chis[count];
        }

        int pairs = 0;
        for (auto& [pure_chie, n] : pure_chis) {
            if (n >= 2) ++pairs;
        }

        // 2つ以上重なっている順子が2個あるならOK
        if (pairs == 2) return 3;

        return std::nullopt;
    }

    std::optional<int> YakuEvaluator::HasSevenPairs(
            const Hand &hand,
            const std::vector<TileTypeCount>& closed_sets,
            const std::vector<TileTypeCount>& opened_sets,
            const std::vector<TileTypeCount>& heads) const noexcept {

        if (heads.size() == 7) return 2;
        return std::nullopt;
    }

    std::optional<int> YakuEvaluator::HasAllPons(
            const Hand &hand,
            const std::vector<TileTypeCount>& closed_sets,
            const std::vector<TileTypeCount>& opened_sets,
            const std::vector<TileTypeCount>& heads) const noexcept {

        if (closed_sets.size() + opened_sets.size() != 4 or heads.size() != 1) {
            // 基本形でなければNG.
            return std::nullopt;
        }

        for (const std::vector<TileTypeCount>& sets : {closed_sets, opened_sets}) {
            for (const TileTypeCount& count : sets) {
                if (count.size() >= 3) {
                    // 順子が含まれるとNG.
                    return std::nullopt;
                }
            }
        }

        return 2;
    }

    std::optional<int> YakuEvaluator::HasPureStraight(
            const Hand &hand,
            const std::vector<TileTypeCount>& closed_sets,
            const std::vector<TileTypeCount>& opened_sets,
            const std::vector<TileTypeCount>& heads) const noexcept {

        if (opened_sets.size() + closed_sets.size() != 4 or heads.size() != 1) {
            // 基本形でなければNG.
            return std::nullopt;
        }

        std::map<TileTypeCount, bool> chis;
        for (const std::vector<TileTypeCount>& sets : {closed_sets, opened_sets}) {
            for (const TileTypeCount &count : sets) {
                if (count.size() == 3) chis[count] = true;
            }
        }

        bool has_straight = false;

        for (int start : {0, 9, 18}) {
            bool valid = true;
            for (int set_start : {start, start + 3, start + 6}) {
                TileTypeCount count;
                for (int i = set_start; i < set_start + 3; ++i) {
                    ++count[static_cast<TileType>(i)];
                }
                if (chis.count(count) == 0) {
                    valid = false;
                    break;
                }
            }
            if (valid) {
                has_straight = true;
                break;
            }
        }

        if (has_straight) {
            if (hand.IsMenzen()) return 2;
            else return 1;
        }

        return std::nullopt;
    }

    std::optional<int> YakuEvaluator::HasMixedTripleChis(
            const Hand &hand,
            const std::vector<TileTypeCount>& closed_sets,
            const std::vector<TileTypeCount>& opened_sets,
            const std::vector<TileTypeCount>& heads) const noexcept {

        if (opened_sets.size() + closed_sets.size() != 4 or heads.size() != 1) {
            // 基本形でなければNG.
            return std::nullopt;
        }

        std::map<TileTypeCount, bool> chis;
        for (const std::vector<TileTypeCount>& sets : {closed_sets, opened_sets}) {
            for (const TileTypeCount &count : sets) {
                if (count.size() == 3) chis[count] = true;
            }
        }

        bool has_mixed_triple_chis = false;

        for (int start = 0; start + 2 < 9; ++start) {
            bool valid = true;
            for (int set_start : {start, start + 9, start + 18}) {
                TileTypeCount count;
                for (int i = set_start; i < set_start + 3; ++i) {
                    ++count[static_cast<TileType>(i)];
                }
                if (chis.count(count) == 0) {
                    valid = false;
                    break;
                }
            }
            if (valid) {
                has_mixed_triple_chis = true;
                break;
            }
        }

        if (has_mixed_triple_chis) {
            if (hand.IsMenzen()) return 2;
            else return 1;
        }

        return std::nullopt;
    }

    std::optional<int> YakuEvaluator::HasTriplePons(
            const Hand &hand,
            const std::vector<TileTypeCount>& closed_sets,
            const std::vector<TileTypeCount>& opened_sets,
            const std::vector<TileTypeCount>& heads) const noexcept {

        if (opened_sets.size() + closed_sets.size() != 4 or heads.size() != 1) {
            // 基本形でなければNG.
            return std::nullopt;
        }

        std::map<TileTypeCount, bool> pons;
        for (const std::vector<TileTypeCount>& sets : {closed_sets, opened_sets}) {
            for (const TileTypeCount &count : sets) {
                if (count.size() == 1) pons[count] = true;
            }
        }

        bool has_triple_pons = false;

        for (int start = 0; start < 9; ++start) {
            bool valid = true;
            for (int set_start : {start, start + 9, start + 18}) {
                TileTypeCount count;
                count[static_cast<TileType>(set_start)] = 3;
                if (pons.count(count) == 0) {
                    valid = false;
                    break;
                }
            }
            if (valid) {
                has_triple_pons = true;
                break;
            }
        }

        if (has_triple_pons) return 2;

        return std::nullopt;
    }

    std::optional<int> YakuEvaluator::HasOutsideHand(
            const Hand &hand,
            const std::vector<TileTypeCount>& closed_sets,
            const std::vector<TileTypeCount>& opened_sets,
            const std::vector<TileTypeCount>& heads) const noexcept {


        bool has_honour = false;
        bool all_yaochu = true;

        for (const std::vector<TileTypeCount>& blocks : {closed_sets, opened_sets, heads}) {
            for (const TileTypeCount& count : blocks) {
                bool valid = false;
                for (auto& [tile_type, _] : count) {
                    if (!has_honour and Is(tile_type, TileSetType::kHonours)) {
                        has_honour = true;
                    }
                    if (Is(tile_type, TileSetType::kYaocyu)) {
                        valid = true;
                        break;
                    } else {
                        all_yaochu = false;
                    }
                }
                if (!valid) {
                    return std::nullopt;
                }
            }
        }

        // 純全帯幺とは重複しない
        if (!has_honour) return std::nullopt;

        // 混老頭とは複合しない
        if (all_yaochu) return std::nullopt;

        if (hand.IsMenzen()) return 2;
        else return 1;
    }

    std::optional<int> YakuEvaluator::HasTerminalsInAllSets(
            const Hand &hand,
            const std::vector<TileTypeCount>& closed_sets,
            const std::vector<TileTypeCount>& opened_sets,
            const std::vector<TileTypeCount>& heads) const noexcept {

        for (const std::vector<TileTypeCount>& blocks : {closed_sets, opened_sets, heads}) {
            for (const TileTypeCount& count : blocks) {
                bool valid = false;
                for (auto& [tile_type, _] : count) {
                    if (Is(tile_type, TileSetType::kTerminals)) {
                        valid = true;
                        break;
                    }
                }
                if (!valid) {
                    return std::nullopt;
                }
            }
        }

        if (hand.IsMenzen()) return 3;
        else return 2;
    }

    std::optional<int> YakuEvaluator::HasFullyConcealdHand(const Hand &hand) const noexcept {
        if (hand.IsMenzen() and hand.Stage() == HandStage::kAfterTsumo) return 1;
        return std::nullopt;
    }

    std::optional<int> YakuEvaluator::HasAllSimples(const TileTypeCount& count) const noexcept {
        for (const auto& [tile_type, _] : count) {
            if (Is(tile_type, TileSetType::kYaocyu)) return std::nullopt;
        }
        return 1;
    }

    std::optional<int> YakuEvaluator::HasWhiteDragon(const TileTypeCount& count) const noexcept {
        if (count.count(TileType::kWD) and count.at(TileType::kWD) >= 3) return 1;
        return std::nullopt;
    }

    std::optional<int> YakuEvaluator::HasGreenDragon(const TileTypeCount& count) const noexcept {
        if (count.count(TileType::kGD) and count.at(TileType::kGD) >= 3) return 1;
        return std::nullopt;
    }

    std::optional<int> YakuEvaluator::HasRedDragon(const TileTypeCount& count) const noexcept {
        if (count.count(TileType::kRD) and count.at(TileType::kRD) >= 3) return 1;
        return std::nullopt;
    }

    std::optional<int> YakuEvaluator::HasAllTermsAndHonours(const TileTypeCount& count) const noexcept {
        for (const auto& [tile_type, _] : count) {
            if (Is(tile_type, TileSetType::kTanyao)) return std::nullopt;
        }
        return 2;
    }

    std::optional<int> YakuEvaluator::HasHalfFlush(const Hand& hand, const TileTypeCount& count) const noexcept {
        std::map<TileSetType,bool> set_types;
        for (const auto& [tile_type, _] : count) {
            if (Is(tile_type, TileSetType::kHonours)) set_types[TileSetType::kHonours] = true;
            else set_types[Color(tile_type)] = true;
        }

        if (set_types.count(TileSetType::kHonours) == 0 or set_types.size() > 2) return std::nullopt;
        if (hand.IsMenzen()) return 3;
        return 2;
    }

    std::optional<int> YakuEvaluator::HasFullFlush(const Hand& hand, const TileTypeCount& count) const noexcept {
        std::map<TileSetType,bool> set_types;
        for (const auto& [tile_type, _] : count) {
            if (Is(tile_type, TileSetType::kHonours)) set_types[TileSetType::kHonours] = true;
            else set_types[Color(tile_type)] = true;
        }

        if (set_types.count(TileSetType::kHonours) or set_types.size() > 1) return std::nullopt;
        if (hand.IsMenzen()) return 6;
        return 5;
    }
}