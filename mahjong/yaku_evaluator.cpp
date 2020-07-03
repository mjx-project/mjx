#include "yaku_evaluator.h"

#include <cassert>
#include <vector>

#include "types.h"
#include "win_score.h"
#include "utils.h"

namespace mj
{
    YakuEvaluator::YakuEvaluator() : win_cache_() {}

    bool YakuEvaluator::Has(const Hand& hand) const noexcept {
        if (win_cache_.Has(ClosedHandTiles(hand))) return true;

        const TileTypeCount all_tiles = ClosedAndOpenedHandTiles(hand);

        return HasThirteenOrphans(hand, all_tiles) or HasCompletedThirteenOrphans(hand, all_tiles);
    }

    std::map<Yaku,int> YakuEvaluator::MaximizeTotalFan(const Hand& hand) const noexcept {

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

        for (const auto& [closed_sets, heads] : win_cache_.SetAndHeads(ClosedHandTiles(hand))) {

            std::map<Yaku,int> yaku_in_this_pattern;

            // 各役の判定
            if (const std::optional<int> fan = HasPinfu(hand, closed_sets, opened_sets, heads); fan) {
                yaku_in_this_pattern[Yaku::kPinfu] = fan.value();
            }
            if (const std::optional<int> fan = HasPureDoubleChis(hand, closed_sets, opened_sets, heads); fan) {
                yaku_in_this_pattern[Yaku::kPureDoubleChis] = fan.value();
            }
            if (const std::optional<int> fan = HasTwicePureDoubleChis(hand, closed_sets, opened_sets, heads); fan) {
                yaku_in_this_pattern[Yaku::kTwicePureDoubleChis] = fan.value();
            }
            if (const std::optional<int> fan = HasSevenPairs(hand, closed_sets, opened_sets, heads); fan) {
                yaku_in_this_pattern[Yaku::kSevenPairs] = fan.value();
            }
            if (const std::optional<int> fan = HasAllPons(hand, closed_sets, opened_sets, heads); fan) {
                yaku_in_this_pattern[Yaku::kAllPons] = fan.value();
            }
            if (const std::optional<int> fan = HasPureStraight(hand, closed_sets, opened_sets, heads); fan) {
                yaku_in_this_pattern[Yaku::kPureStraight] = fan.value();
            }
            if (const std::optional<int> fan = HasMixedTripleChis(hand, closed_sets, opened_sets, heads); fan) {
                yaku_in_this_pattern[Yaku::kMixedTripleChis] = fan.value();
            }
            if (const std::optional<int> fan = HasTriplePons(hand, closed_sets, opened_sets, heads); fan) {
                yaku_in_this_pattern[Yaku::kTriplePons] = fan.value();
            }
            if (const std::optional<int> fan = HasOutsideHand(hand, closed_sets, opened_sets, heads); fan) {
                yaku_in_this_pattern[Yaku::kOutsideHand] = fan.value();
            }
            if (const std::optional<int> fan = HasTerminalsInAllSets(hand, closed_sets, opened_sets, heads); fan) {
                yaku_in_this_pattern[Yaku::kTerminalsInAllSets] = fan.value();
            }

            // 今までに調べた組み合わせ方より役の総得点が高いなら採用する.
            if (TotalFan(best_yaku) < TotalFan(yaku_in_this_pattern)) {
                std::swap(best_yaku, yaku_in_this_pattern);
            }
        }

        return best_yaku;
    }

    WinningScore YakuEvaluator::Eval(const Hand& hand) const noexcept {

        assert(Has(hand));

        const TileTypeCount all_tiles = ClosedAndOpenedHandTiles(hand);

        WinningScore score;

        // 役満の判定
        if (HasBigThreeDragons(all_tiles)) {
            score.AddYakuman(Yaku::kBigThreeDragons);
        }
        if (HasAllHonours(all_tiles)) {
            score.AddYakuman(Yaku::kAllHonours);
        }
        if (HasAllGreen(all_tiles)) {
            score.AddYakuman(Yaku::kAllGreen);
        }
        if (HasAllTerminals(all_tiles)) {
            score.AddYakuman(Yaku::kAllTerminals);
        }
        if (HasBigFourWinds(all_tiles)) {
            score.AddYakuman(Yaku::kBigFourWinds);
        }
        if (HasLittleFourWinds(all_tiles)) {
            score.AddYakuman(Yaku::kLittleFourWinds);
        }
        if (HasThirteenOrphans(hand, all_tiles)) {
            score.AddYakuman(Yaku::kThirteenOrphans);
        }
        if (HasCompletedThirteenOrphans(hand, all_tiles)) {
            score.AddYakuman(Yaku::kCompletedThirteenOrphans);
        }
        if (HasNineGates(hand, all_tiles)) {
            score.AddYakuman(Yaku::kNineGates);
        }
        if (HasPureNineGates(hand, all_tiles)) {
            score.AddYakuman(Yaku::kPureNineGates);
        }
        if (HasFourKans(hand)) {
            score.AddYakuman(Yaku::kFourKans);
        }
        if (HasFourConcealedPons(hand, all_tiles)) {
            score.AddYakuman(Yaku::kFourConcealedPons);
        }
        if (HasCompletedFourConcealedPons(hand, all_tiles)) {
            score.AddYakuman(Yaku::kCompletedFourConcealedPons);
        }

        if (!score.RequireFan()) return score;

        // 手牌の組み合わせ方によらない役
        if (const std::optional<int> fan = HasFullyConcealdHand(hand); fan) {
            score.AddYaku(Yaku::kFullyConcealedHand, fan.value());
        }
        if (const std::optional<int> fan = HasAllSimples(all_tiles); fan) {
            score.AddYaku(Yaku::kAllSimples, fan.value());
        }
        if (const std::optional<int> fan = HasWhiteDragon(all_tiles); fan) {
            score.AddYaku(Yaku::kWhiteDragon, fan.value());
        }
        if (const std::optional<int> fan = HasGreenDragon(all_tiles); fan) {
            score.AddYaku(Yaku::kGreenDragon, fan.value());
        }
        if (const std::optional<int> fan = HasRedDragon(all_tiles); fan) {
            score.AddYaku(Yaku::kRedDragon, fan.value());
        }
        if (const std::optional<int> fan = HasAllTermsAndHonours(all_tiles); fan) {
            score.AddYaku(Yaku::kAllTermsAndHonours, fan.value());
        }
        if (const std::optional<int> fan = HasHalfFlush(hand, all_tiles); fan) {
            score.AddYaku(Yaku::kHalfFlush, fan.value());
        }
        if (const std::optional<int> fan = HasFullFlush(hand, all_tiles); fan) {
            score.AddYaku(Yaku::kFullFlush, fan.value());
        }
        if (const std::optional<int> fan = HasThreeKans(hand); fan) {
            score.AddYaku(Yaku::kThreeKans, fan.value());
        }

        // 手牌の組み合わせ方に依存する役
        const std::map<Yaku,int> best_yaku = MaximizeTotalFan(hand);

        for (auto& [yaku, fan] : best_yaku) score.AddYaku(yaku, fan);

        if (!score.RequireFu()) return score;

        // TODO calculate fu;
        return score;
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
            const std::vector<TileTypeCount>& heads) noexcept {
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
            const std::vector<TileTypeCount>& heads) noexcept {

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
        for (const auto& [pure_chie, n] : pure_chis) {
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
            const std::vector<TileTypeCount>& heads) noexcept {

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
        for (const auto& [pure_chie, n] : pure_chis) {
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
            const std::vector<TileTypeCount>& heads) noexcept {

        if (heads.size() == 7) return 2;
        return std::nullopt;
    }

    std::optional<int> YakuEvaluator::HasAllPons(
            const Hand &hand,
            const std::vector<TileTypeCount>& closed_sets,
            const std::vector<TileTypeCount>& opened_sets,
            const std::vector<TileTypeCount>& heads) noexcept {

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
            const std::vector<TileTypeCount>& heads) noexcept {

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
            const std::vector<TileTypeCount>& heads) noexcept {

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
            const std::vector<TileTypeCount>& heads) noexcept {

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
            const std::vector<TileTypeCount>& heads) noexcept {


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

        // 純全帯幺とは複合しない
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
            const std::vector<TileTypeCount>& heads) noexcept {

        for (const std::vector<TileTypeCount>& blocks : {closed_sets, opened_sets, heads}) {
            for (const TileTypeCount& count : blocks) {
                bool valid = false;
                for (const auto& [tile_type, _] : count) {
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

    std::optional<int> YakuEvaluator::HasFullyConcealdHand(const Hand &hand) noexcept {
        if (hand.IsMenzen() and hand.Stage() == HandStage::kAfterTsumo) return 1;
        return std::nullopt;
    }

    std::optional<int> YakuEvaluator::HasAllSimples(const TileTypeCount& count) noexcept {
        for (const auto& [tile_type, _] : count) {
            if (Is(tile_type, TileSetType::kYaocyu)) return std::nullopt;
        }
        return 1;
    }

    std::optional<int> YakuEvaluator::HasWhiteDragon(const TileTypeCount& count) noexcept {
        if (count.count(TileType::kWD) and count.at(TileType::kWD) >= 3) return 1;
        return std::nullopt;
    }

    std::optional<int> YakuEvaluator::HasGreenDragon(const TileTypeCount& count) noexcept {
        if (count.count(TileType::kGD) and count.at(TileType::kGD) >= 3) return 1;
        return std::nullopt;
    }

    std::optional<int> YakuEvaluator::HasRedDragon(const TileTypeCount& count) noexcept {
        if (count.count(TileType::kRD) and count.at(TileType::kRD) >= 3) return 1;
        return std::nullopt;
    }

    std::optional<int> YakuEvaluator::HasAllTermsAndHonours(const TileTypeCount& count) noexcept {
        for (const auto& [tile_type, _] : count) {
            if (Is(tile_type, TileSetType::kTanyao)) return std::nullopt;
        }
        return 2;
    }

    std::optional<int> YakuEvaluator::HasHalfFlush(const Hand& hand, const TileTypeCount& count) noexcept {
        std::map<TileSetType,bool> set_types;
        for (const auto& [tile_type, _] : count) {
            if (Is(tile_type, TileSetType::kHonours)) set_types[TileSetType::kHonours] = true;
            else set_types[Color(tile_type)] = true;
        }

        if (set_types.count(TileSetType::kHonours) == 0 or set_types.size() > 2) return std::nullopt;
        if (hand.IsMenzen()) return 3;
        return 2;
    }

    std::optional<int> YakuEvaluator::HasFullFlush(const Hand& hand, const TileTypeCount& count) noexcept {
        std::map<TileSetType,bool> set_types;
        for (const auto& [tile_type, _] : count) {
            if (Is(tile_type, TileSetType::kHonours)) set_types[TileSetType::kHonours] = true;
            else set_types[Color(tile_type)] = true;
        }

        if (set_types.count(TileSetType::kHonours) or set_types.size() > 1) return std::nullopt;
        if (hand.IsMenzen()) return 6;
        return 5;
    }

    std::optional<int> YakuEvaluator::HasThreeKans(const Hand& hand) noexcept {
        int kans = 0;
        for (const Open* open : hand.Opens()) {
            if (any_of(open->Type(), {
                    OpenType::kKanOpened, OpenType::kKanAdded, OpenType::kKanClosed})) {
                ++kans;
            }
        }

        if (kans < 3) return std::nullopt;
        return 2;
    }

    bool YakuEvaluator::HasBigThreeDragons(const TileTypeCount& count) noexcept {
        return count.count(TileType::kWD) and count.at(TileType::kWD) >= 3 and
               count.count(TileType::kGD) and count.at(TileType::kGD) >= 3 and
               count.count(TileType::kRD) and count.at(TileType::kRD) >= 3;
    }

    bool YakuEvaluator::HasAllHonours(const TileTypeCount& count) noexcept {
        for (const auto& [tile_type, _] : count) {
            if (!Is(tile_type, TileSetType::kHonours)) {
                return false;
            }
        }
        return true;
    }

    bool YakuEvaluator::HasAllGreen(const TileTypeCount& count) noexcept {
        for (const auto& [tile_type, _] : count) {
            if (!Is(tile_type, TileSetType::kGreen)) {
                return false;
            }
        }
        return true;
    }

    bool YakuEvaluator::HasAllTerminals(const TileTypeCount& count) noexcept {
        for (const auto& [tile_type, _] : count) {
            if (!Is(tile_type, TileSetType::kTerminals)) {
                return false;
            }
        }
        return true;
    }

    bool YakuEvaluator::HasBigFourWinds(const TileTypeCount& count) noexcept {
        return count.count(TileType::kEW) and count.at(TileType::kEW) >= 3 and
               count.count(TileType::kSW) and count.at(TileType::kSW) >= 3 and
               count.count(TileType::kWW) and count.at(TileType::kWW) >= 3 and
               count.count(TileType::kNW) and count.at(TileType::kNW) >= 3;
    }

    bool YakuEvaluator::HasLittleFourWinds(const TileTypeCount& count) noexcept {
        int pons = 0, heads = 0;
        for (const TileType tile_type : {TileType::kEW, TileType::kSW, TileType::kWW, TileType::kNW}) {
            if (!count.count(tile_type)) return false;
            if (count.at(tile_type) >= 3) ++pons;
            else if (count.at(tile_type) == 2) ++heads;
        }
        return pons == 3 and heads == 1;
    }

    bool YakuEvaluator::HasThirteenOrphans(const Hand& hand, const TileTypeCount& count) noexcept {
        std::map<TileType, int> yaocyu;
        for (const auto& [tile_type, n] : count) {
            if (Is(tile_type, TileSetType::kYaocyu)) {
                yaocyu[tile_type] = n;
            }
        }
        assert(hand.LastTileAdded().has_value());
        const Tile tsumo = hand.LastTileAdded().value();

        return yaocyu.size() == 13 and yaocyu[tsumo.Type()] == 1;
    }

    bool YakuEvaluator::HasCompletedThirteenOrphans(const Hand& hand, const TileTypeCount& count) noexcept {
        TileTypeCount yaocyu;
        for (const auto& [tile_type, n] : count) {
            if (Is(tile_type, TileSetType::kYaocyu)) {
                yaocyu[tile_type] = n;
            }
        }
        assert(hand.LastTileAdded().has_value());
        const Tile tsumo = hand.LastTileAdded().value();

        return yaocyu.size() == 13 and yaocyu[tsumo.Type()] == 2;
    }

    bool YakuEvaluator::HasNineGates(const Hand& hand, const TileTypeCount& count) noexcept {
        if (!hand.IsMenzen()) return false;
        std::map<TileSetType, bool> colors;
        for (const auto& [tile_type, n] : count) {
            if (Is(tile_type, TileSetType::kHonours)) return false;
            colors[Color(tile_type)] = true;
        }
        if (colors.size() > 1) return false;

        std::vector<int> required{0,3,1,1,1,1,1,1,1,3};

        assert(hand.LastTileAdded().has_value());
        const Tile tsumo = hand.LastTileAdded().value();

        for (const auto& [tile_type, n] : count) {
            if (required[Num(tile_type)] > n) return false;
            if (required[Num(tile_type)] < n and tile_type == tsumo.Type()) return false;
        }

        return true;
    }

    bool YakuEvaluator::HasPureNineGates(const Hand& hand, const TileTypeCount& count) noexcept {
        if (!hand.IsMenzen()) return false;
        std::map<TileSetType, bool> colors;
        for (const auto& [tile_type, n] : count) {
            if (Is(tile_type, TileSetType::kHonours)) return false;
            colors[Color(tile_type)] = true;
        }
        if (colors.size() > 1) return false;

        std::vector<int> required{0,3,1,1,1,1,1,1,1,3};

        assert(hand.LastTileAdded().has_value());
        const Tile tsumo = hand.LastTileAdded().value();

        for (const auto& [tile_type, n] : count) {
            if (required[Num(tile_type)] > n) return false;
            if (required[Num(tile_type)] == n and tile_type == tsumo.Type()) return false;
        }

        return true;
    }

    bool YakuEvaluator::HasFourKans(const Hand& hand) noexcept {
        int kans = 0;
        for (const Open* open : hand.Opens()) {
            if (any_of(open->Type(), {
                    OpenType::kKanOpened, OpenType::kKanAdded, OpenType::kKanClosed})) {
                ++kans;
            }
        }

        return kans == 4;
    }

    bool YakuEvaluator::HasFourConcealedPons(const Hand& hand, const TileTypeCount& count) noexcept {
        if (!hand.IsMenzen()) return false;
        if (count.size() != 5) return false;

        assert(hand.LastTileAdded().has_value());
        const Tile tsumo = hand.LastTileAdded().value();

        return count.at(tsumo.Type()) > 2;
    }

    bool YakuEvaluator::HasCompletedFourConcealedPons(const Hand& hand, const TileTypeCount& count) noexcept {
        if (!hand.IsMenzen()) return false;
        if (count.size() != 5) return false;

        assert(hand.LastTileAdded().has_value());
        const Tile tsumo = hand.LastTileAdded().value();

        return count.at(tsumo.Type()) == 2;
    }
}