#include "yaku_evaluator.h"

#include <cassert>
#include <vector>
#include <tuple>

#include "types.h"
#include "win_score.h"
#include "utils.h"

namespace mj
{
    const WinningHandCache &YakuEvaluator::win_cache() {
        return WinningHandCache::instance();
    }

    bool YakuEvaluator::Has(const WinningInfo& win_info) noexcept {
        WinningScore score;

        // closedな手牌が上がり形になっている or 国士無双かどうかを判定する.
        return win_cache().Has(win_info.closed_tile_types) or
               HasThirteenOrphans(win_info) or
               HasCompletedThirteenOrphans(win_info);
    }

    bool YakuEvaluator::CanWin(const WinningInfo& win_info) noexcept {
        WinningScore score;

        // closedな手牌が上がり形になっている or 国士無双かどうかを判定する.
        if (!win_cache().Has(win_info.closed_tile_types) and
            !HasThirteenOrphans(win_info) and
            !HasCompletedThirteenOrphans(win_info)
                ) return false;

        // 役満の判定
        JudgeYakuman(win_info, score);
        if (!score.yakuman().empty()) return true;

        // 手牌の組み合わせ方によらない役
        JudgeSimpleYaku(win_info, score);
        if (!score.yaku().empty()) return true;

        // 手牌の組み合わせ方に依存する役
        const auto [best_yaku, closed_sets, heads] = MaximizeTotalFan(win_info);
        for (auto& [yaku, fan] : best_yaku) score.AddYaku(yaku, fan);
//  TODO: Yakuがdoraのみの場合をはじく
        return !score.yaku().empty();
    }

    WinningScore YakuEvaluator::Eval(const WinningInfo& win_info) noexcept {

        assert(Has(win_info));

        WinningScore score;

        // 役満の判定
        JudgeYakuman(win_info, score);
        if (!score.RequireFan()) return score;

        // 手牌の組み合わせ方によらない役
        JudgeSimpleYaku(win_info, score);

        // 手牌の組み合わせ方に依存する役
        const auto [best_yaku, closed_sets, heads] = MaximizeTotalFan(win_info);
        for (auto& [yaku, fan] : best_yaku) score.AddYaku(yaku, fan);

        // TODO: 役がないと上がれない.
        assert(!score.yaku().empty());

        if (!score.RequireFu()) return score;

        // 符を計算する
        score.SetFu(CalculateFu(win_info, closed_sets, heads, score));

        return score;
    }

    int YakuEvaluator::CalculateFu(
            const WinningInfo& win_info,
            const std::vector<TileTypeCount>& closed_sets,
            const std::vector<TileTypeCount>& heads,
            const WinningScore& win_score) noexcept {

        // 七対子:25
        if (win_score.HasYaku(Yaku::kSevenPairs)) {
            return 25;
        }

        // 平和ツモ:20, 平和ロン:30
        if (win_score.HasYaku(Yaku::kPinfu)) {
            if (win_info.stage == HandStage::kAfterTsumo) {
                return 20;
            } else if (win_info.stage == HandStage::kAfterRon) {
                return 30;
            } else {
                assert(false);
            }
        }

        int fu = 20;

        // 面子
        for (const std::unique_ptr<Open>& open : win_info.opens) {
            OpenType open_type = open->Type();
            if (open_type == OpenType::kChi) continue;

            bool is_yaocyu = Is(open->At(0).Type(), TileSetType::kYaocyu);

            switch (open->Type()) {
                case OpenType::kPon:
                    fu += is_yaocyu ? 4 : 2;
                    break;
                case OpenType::kKanOpened:
                    fu += is_yaocyu ? 16 : 8;
                    break;
                case OpenType::kKanAdded:
                    fu += is_yaocyu ? 16 : 8;
                    break;
                case OpenType::kKanClosed:
                    fu += is_yaocyu ? 32 : 16;
                    break;
                case OpenType::kChi:
                    assert(false);
            }
        }
        for (const auto& closed_set : closed_sets) {
            if (closed_set.size() > 1) continue;    // 順子は0
            bool is_yaocyu = Is(closed_set.begin()->first, TileSetType::kYaocyu);
            fu += is_yaocyu ? 8 : 4;    // 暗刻
        }

        // 雀頭
        assert(heads.size() == 1);
        TileType head_type = heads[0].begin()->first;
        if (Is(head_type, TileSetType::kDragons)) fu += 2;
        // TODO: 場風,自風は2符.
        // TODO: 連風牌は2符? 4符? 要確認.

        // 待ち
        bool has_bad_machi = false;
        assert(win_info.last_added_tile_type);
        const TileType tsumo_type = win_info.last_added_tile_type.value();
        if (tsumo_type == head_type) has_bad_machi = true;    // 単騎
        for (const TileTypeCount& st : closed_sets) {
            if (st.size() == 1) continue;   // 刻子は弾く
            assert(st.size() == 3);
            auto it = st.begin();
            const TileType left = it->first; ++it;
            const TileType center = it->first; ++it;
            const TileType right = it->first;

            if (tsumo_type == center) has_bad_machi = true;     // カンチャン
            if ((tsumo_type == left and Num(right) == 9) or
                (tsumo_type == right and Num(left) == 1)) {
                has_bad_machi = true;                           // ペンチャン
            }
        }
        if (has_bad_machi) fu += 2;

        // 面前加符
        if (win_info.is_menzen and win_info.stage == HandStage::kAfterRon) {
            fu += 10;
        }

        // ツモ符
        if (win_info.stage == HandStage::kAfterTsumo) {
            fu += 2;
        }

        if (fu == 20) {
            // 喰い平和でも最低30符
            return 30;
        }

        // 切り上げ
        if (fu % 10) fu += 10 - fu % 10;
        return fu;
    }

    std::tuple<std::map<Yaku,int>,std::vector<TileTypeCount>,std::vector<TileTypeCount>>
    YakuEvaluator::MaximizeTotalFan(const WinningInfo& win_info) noexcept {

        std::map<Yaku,int> best_yaku;
        std::vector<TileTypeCount> best_closed_set, best_heads;

        std::vector<TileTypeCount> opened_sets;
        for (const std::unique_ptr<Open>& open : win_info.opens) {
            TileTypeCount count;
            for (const Tile tile : open->Tiles()) {
                ++count[tile.Type()];
            }
            opened_sets.push_back(count);
        }

        for (const auto& [closed_sets, heads] : win_cache().SetAndHeads(win_info.closed_tile_types)) {

            std::map<Yaku,int> yaku_in_this_pattern;

            // 各役の判定
            if (const std::optional<int> fan = HasPinfu(win_info, closed_sets, opened_sets, heads); fan) {
                yaku_in_this_pattern[Yaku::kPinfu] = fan.value();
            }
            if (const std::optional<int> fan = HasPureDoubleChis(win_info, closed_sets, opened_sets, heads); fan) {
                yaku_in_this_pattern[Yaku::kPureDoubleChis] = fan.value();
            }
            if (const std::optional<int> fan = HasTwicePureDoubleChis(win_info, closed_sets, opened_sets, heads); fan) {
                yaku_in_this_pattern[Yaku::kTwicePureDoubleChis] = fan.value();
            }
            if (const std::optional<int> fan = HasSevenPairs(win_info, closed_sets, opened_sets, heads); fan) {
                yaku_in_this_pattern[Yaku::kSevenPairs] = fan.value();
            }
            if (const std::optional<int> fan = HasAllPons(win_info, closed_sets, opened_sets, heads); fan) {
                yaku_in_this_pattern[Yaku::kAllPons] = fan.value();
            }
            if (const std::optional<int> fan = HasPureStraight(win_info, closed_sets, opened_sets, heads); fan) {
                yaku_in_this_pattern[Yaku::kPureStraight] = fan.value();
            }
            if (const std::optional<int> fan = HasMixedTripleChis(win_info, closed_sets, opened_sets, heads); fan) {
                yaku_in_this_pattern[Yaku::kMixedTripleChis] = fan.value();
            }
            if (const std::optional<int> fan = HasTriplePons(win_info, closed_sets, opened_sets, heads); fan) {
                yaku_in_this_pattern[Yaku::kTriplePons] = fan.value();
            }
            if (const std::optional<int> fan = HasOutsideHand(win_info, closed_sets, opened_sets, heads); fan) {
                yaku_in_this_pattern[Yaku::kOutsideHand] = fan.value();
            }
            if (const std::optional<int> fan = HasTerminalsInAllSets(win_info, closed_sets, opened_sets, heads); fan) {
                yaku_in_this_pattern[Yaku::kTerminalsInAllSets] = fan.value();
            }

            // 今までに調べた組み合わせ方より役の総飜数が高いなら採用する.
            if (best_yaku.empty() or TotalFan(best_yaku) < TotalFan(yaku_in_this_pattern)) {
                std::swap(best_yaku, yaku_in_this_pattern);
                best_closed_set = closed_sets;
                best_heads = heads;
            }
        }

        return {best_yaku, best_closed_set, best_heads};
    }

    void YakuEvaluator::JudgeYakuman(
            const WinningInfo& win_info,
            WinningScore& score) noexcept {

        if (HasBigThreeDragons(win_info)) {
            score.AddYakuman(Yaku::kBigThreeDragons);
        }
        if (HasAllHonours(win_info)) {
            score.AddYakuman(Yaku::kAllHonours);
        }
        if (HasAllGreen(win_info)) {
            score.AddYakuman(Yaku::kAllGreen);
        }
        if (HasAllTerminals(win_info)) {
            score.AddYakuman(Yaku::kAllTerminals);
        }
        if (HasBigFourWinds(win_info)) {
            score.AddYakuman(Yaku::kBigFourWinds);
        }
        if (HasLittleFourWinds(win_info)) {
            score.AddYakuman(Yaku::kLittleFourWinds);
        }
        if (HasThirteenOrphans(win_info)) {
            score.AddYakuman(Yaku::kThirteenOrphans);
        }
        if (HasCompletedThirteenOrphans(win_info)) {
            score.AddYakuman(Yaku::kCompletedThirteenOrphans);
        }
        if (HasNineGates(win_info)) {
            score.AddYakuman(Yaku::kNineGates);
        }
        if (HasPureNineGates(win_info)) {
            score.AddYakuman(Yaku::kPureNineGates);
        }
        if (HasFourKans(win_info)) {
            score.AddYakuman(Yaku::kFourKans);
        }
        if (HasFourConcealedPons(win_info)) {
            score.AddYakuman(Yaku::kFourConcealedPons);
        }
        if (HasCompletedFourConcealedPons(win_info)) {
            score.AddYakuman(Yaku::kCompletedFourConcealedPons);
        }

    }

    void YakuEvaluator::JudgeSimpleYaku(
            const WinningInfo& win_info,
            WinningScore& score) noexcept {

        if (const std::optional<int> fan = HasFullyConcealdHand(win_info); fan) {
            score.AddYaku(Yaku::kFullyConcealedHand, fan.value());
        }
        if (const std::optional<int> fan = HasRiichi(win_info); fan) {
            score.AddYaku(Yaku::kRiichi, fan.value());
        }
        if (const std::optional<int> fan = HasAllSimples(win_info); fan) {
            score.AddYaku(Yaku::kAllSimples, fan.value());
        }
        if (const std::optional<int> fan = HasWhiteDragon(win_info); fan) {
            score.AddYaku(Yaku::kWhiteDragon, fan.value());
        }
        if (const std::optional<int> fan = HasGreenDragon(win_info); fan) {
            score.AddYaku(Yaku::kGreenDragon, fan.value());
        }
        if (const std::optional<int> fan = HasRedDragon(win_info); fan) {
            score.AddYaku(Yaku::kRedDragon, fan.value());
        }
        if (const std::optional<int> fan = HasSeatWindEast(win_info); fan) {
            score.AddYaku(Yaku::kSeatWindEast, fan.value());
        }
        if (const std::optional<int> fan = HasSeatWindSouth(win_info); fan) {
            score.AddYaku(Yaku::kSeatWindSouth, fan.value());
        }
        if (const std::optional<int> fan = HasSeatWindWest(win_info); fan) {
            score.AddYaku(Yaku::kSeatWindWest, fan.value());
        }
        if (const std::optional<int> fan = HasSeatWindNorth(win_info); fan) {
            score.AddYaku(Yaku::kSeatWindNorth, fan.value());
        }
        if (const std::optional<int> fan = HasPrevalentWindEast(win_info); fan) {
            score.AddYaku(Yaku::kPrevalentWindEast, fan.value());
        }
        if (const std::optional<int> fan = HasPrevalentWindSouth(win_info); fan) {
            score.AddYaku(Yaku::kPrevalentWindSouth, fan.value());
        }
        if (const std::optional<int> fan = HasPrevalentWindWest(win_info); fan) {
            score.AddYaku(Yaku::kPrevalentWindWest, fan.value());
        }
        if (const std::optional<int> fan = HasPrevalentWindNorth(win_info); fan) {
            score.AddYaku(Yaku::kPrevalentWindNorth, fan.value());
        }
        if (const std::optional<int> fan = HasAllTermsAndHonours(win_info); fan) {
            score.AddYaku(Yaku::kAllTermsAndHonours, fan.value());
        }
        if (const std::optional<int> fan = HasHalfFlush(win_info); fan) {
            score.AddYaku(Yaku::kHalfFlush, fan.value());
        }
        if (const std::optional<int> fan = HasFullFlush(win_info); fan) {
            score.AddYaku(Yaku::kFullFlush, fan.value());
        }
        if (const std::optional<int> fan = HasThreeKans(win_info); fan) {
            score.AddYaku(Yaku::kThreeKans, fan.value());
        }
        if (const std::optional<int> fan = HasLittleThreeDragons(win_info); fan) {
            score.AddYaku(Yaku::kLittleThreeDragons, fan.value());
        }
    }

    int YakuEvaluator::TotalFan(const std::map<Yaku,int>& yaku) noexcept {
        int total = 0;
        for (const auto& [key, val] : yaku) total += val;
        return total;
    }

    std::optional<int> YakuEvaluator::HasPinfu(
            const WinningInfo& win_info,
            const std::vector<TileTypeCount>& closed_sets,
            const std::vector<TileTypeCount>& opened_sets,
            const std::vector<TileTypeCount>& heads) noexcept {
        if (!win_info.is_menzen) return std::nullopt;

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

        assert(win_info.last_added_tile_type);
        const TileType tsumo_type = win_info.last_added_tile_type.value();

        for (const TileTypeCount& st : closed_sets) {
            const TileType left = st.begin()->first,
                           right = st.rbegin()->first;
            if ((tsumo_type == left and Num(right) != 9) or
                (tsumo_type == right and Num(left) != 1)) {

                // いずれかの順子のリャンメン待ちをツモっていたらOK
                return 1;
            }
        }

        // リャンメン待ちでなければNG
        return std::nullopt;
    }

    std::optional<int> YakuEvaluator::HasPureDoubleChis(
            const WinningInfo& win_info,
            const std::vector<TileTypeCount>& closed_sets,
            const std::vector<TileTypeCount>& opened_sets,
            const std::vector<TileTypeCount>& heads) noexcept {

        // 鳴いているとNG
        if (!win_info.is_menzen) return std::nullopt;

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
            const WinningInfo& win_info,
            const std::vector<TileTypeCount>& closed_sets,
            const std::vector<TileTypeCount>& opened_sets,
            const std::vector<TileTypeCount>& heads) noexcept {

        // 鳴いているとNG
        if (!win_info.is_menzen) return std::nullopt;

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
            const WinningInfo& win_info,
            const std::vector<TileTypeCount>& closed_sets,
            const std::vector<TileTypeCount>& opened_sets,
            const std::vector<TileTypeCount>& heads) noexcept {

        if (heads.size() == 7) return 2;
        return std::nullopt;
    }

    std::optional<int> YakuEvaluator::HasAllPons(
            const WinningInfo& win_info,
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
            const WinningInfo& win_info,
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
            if (win_info.is_menzen) return 2;
            else return 1;
        }

        return std::nullopt;
    }

    std::optional<int> YakuEvaluator::HasMixedTripleChis(
            const WinningInfo& win_info,
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
            if (win_info.is_menzen) return 2;
            else return 1;
        }

        return std::nullopt;
    }

    std::optional<int> YakuEvaluator::HasTriplePons(
            const WinningInfo& win_info,
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
            const WinningInfo& win_info,
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

        if (win_info.is_menzen) return 2;
        else return 1;
    }

    std::optional<int> YakuEvaluator::HasTerminalsInAllSets(
            const WinningInfo& win_info,
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

        if (win_info.is_menzen) return 3;
        else return 2;
    }

    std::optional<int> YakuEvaluator::HasFullyConcealdHand(const WinningInfo& win_info) noexcept {
        if (win_info.is_menzen and win_info.stage == HandStage::kAfterTsumo) return 1;
        return std::nullopt;
    }

    std::optional<int> YakuEvaluator::HasRiichi(const WinningInfo& win_info) noexcept {
        if (win_info.under_riichi) return 1;
        return std::nullopt;
    }

    std::optional<int> YakuEvaluator::HasAllSimples(const WinningInfo& win_info) noexcept {
        for (const auto& [tile_type, _] : win_info.all_tile_types) {
            if (Is(tile_type, TileSetType::kYaocyu)) return std::nullopt;
        }
        return 1;
    }

    std::optional<int> YakuEvaluator::HasWhiteDragon(const WinningInfo& win_info) noexcept {
        const auto& all_tile_types = win_info.all_tile_types;
        if (all_tile_types.count(TileType::kWD) and all_tile_types.at(TileType::kWD) >= 3) return 1;
        return std::nullopt;
    }

    std::optional<int> YakuEvaluator::HasGreenDragon(const WinningInfo& win_info) noexcept {
        const auto& all_tile_types = win_info.all_tile_types;
        if (all_tile_types.count(TileType::kGD) and all_tile_types.at(TileType::kGD) >= 3) return 1;
        return std::nullopt;
    }

    std::optional<int> YakuEvaluator::HasRedDragon(const WinningInfo& win_info) noexcept {
        const auto& all_tile_types = win_info.all_tile_types;
        if (all_tile_types.count(TileType::kRD) and all_tile_types.at(TileType::kRD) >= 3) return 1;
        return std::nullopt;
    }

    std::optional<int> YakuEvaluator::HasSeatWindEast(const WinningInfo& win_info) noexcept {
        if (win_info.seat_wind != Wind::kEast) return std::nullopt;
        const auto& all_tile_types = win_info.all_tile_types;
        if (all_tile_types.count(TileType::kEW) and all_tile_types.at(TileType::kEW) >= 3) return 1;
        return std::nullopt;
    }

    std::optional<int> YakuEvaluator::HasSeatWindSouth(const WinningInfo& win_info) noexcept {
        if (win_info.seat_wind != Wind::kSouth) return std::nullopt;
        const auto& all_tile_types = win_info.all_tile_types;
        if (all_tile_types.count(TileType::kSW) and all_tile_types.at(TileType::kSW) >= 3) return 1;
        return std::nullopt;
    }

    std::optional<int> YakuEvaluator::HasSeatWindWest(const WinningInfo& win_info) noexcept {
        if (win_info.seat_wind != Wind::kWest) return std::nullopt;
        const auto& all_tile_types = win_info.all_tile_types;
        if (all_tile_types.count(TileType::kWW) and all_tile_types.at(TileType::kWW) >= 3) return 1;
        return std::nullopt;
    }

    std::optional<int> YakuEvaluator::HasSeatWindNorth(const WinningInfo& win_info) noexcept {
        if (win_info.seat_wind != Wind::kNorth) return std::nullopt;
        const auto& all_tile_types = win_info.all_tile_types;
        if (all_tile_types.count(TileType::kNW) and all_tile_types.at(TileType::kNW) >= 3) return 1;
        return std::nullopt;
    }

    std::optional<int> YakuEvaluator::HasPrevalentWindEast(const WinningInfo& win_info) noexcept {
        if (win_info.prevalent_wind != Wind::kEast) return std::nullopt;
        const auto& all_tile_types = win_info.all_tile_types;
        if (all_tile_types.count(TileType::kEW) and all_tile_types.at(TileType::kEW) >= 3) return 1;
        return std::nullopt;
    }

    std::optional<int> YakuEvaluator::HasPrevalentWindSouth(const WinningInfo& win_info) noexcept {
        if (win_info.prevalent_wind != Wind::kSouth) return std::nullopt;
        const auto& all_tile_types = win_info.all_tile_types;
        if (all_tile_types.count(TileType::kSW) and all_tile_types.at(TileType::kSW) >= 3) return 1;
        return std::nullopt;
    }

    std::optional<int> YakuEvaluator::HasPrevalentWindWest(const WinningInfo& win_info) noexcept {
        if (win_info.prevalent_wind != Wind::kWest) return std::nullopt;
        const auto& all_tile_types = win_info.all_tile_types;
        if (all_tile_types.count(TileType::kWW) and all_tile_types.at(TileType::kWW) >= 3) return 1;
        return std::nullopt;
    }

    std::optional<int> YakuEvaluator::HasPrevalentWindNorth(const WinningInfo& win_info) noexcept {
        if (win_info.prevalent_wind != Wind::kNorth) return std::nullopt;
        const auto& all_tile_types = win_info.all_tile_types;
        if (all_tile_types.count(TileType::kNW) and all_tile_types.at(TileType::kNW) >= 3) return 1;
        return std::nullopt;
    }

    std::optional<int> YakuEvaluator::HasAllTermsAndHonours(const WinningInfo& win_info) noexcept {
        const auto& all_tile_types = win_info.all_tile_types;
        for (const auto& [tile_type, _] : all_tile_types) {
            if (Is(tile_type, TileSetType::kTanyao)) return std::nullopt;
        }
        return 2;
    }

    std::optional<int> YakuEvaluator::HasHalfFlush(const WinningInfo& win_info) noexcept {
        const auto& all_tile_types = win_info.all_tile_types;
        std::map<TileSetType,bool> set_types;
        for (const auto& [tile_type, _] : all_tile_types) {
            if (Is(tile_type, TileSetType::kHonours)) set_types[TileSetType::kHonours] = true;
            else set_types[Color(tile_type)] = true;
        }

        if (set_types.count(TileSetType::kHonours) == 0 or set_types.size() > 2) return std::nullopt;
        if (win_info.is_menzen) return 3;
        return 2;
    }

    std::optional<int> YakuEvaluator::HasFullFlush(const WinningInfo& win_info) noexcept {
        const auto& all_tile_types = win_info.all_tile_types;
        std::map<TileSetType,bool> set_types;
        for (const auto& [tile_type, _] : all_tile_types) {
            if (Is(tile_type, TileSetType::kHonours)) set_types[TileSetType::kHonours] = true;
            else set_types[Color(tile_type)] = true;
        }

        if (set_types.count(TileSetType::kHonours) or set_types.size() > 1) return std::nullopt;
        if (win_info.is_menzen) return 6;
        return 5;
    }

    std::optional<int> YakuEvaluator::HasThreeKans(const WinningInfo& win_info) noexcept {
        int kans = 0;
        for (const std::unique_ptr<Open>& open : win_info.opens) {
            if (Any(open->Type(), {
                    OpenType::kKanOpened, OpenType::kKanAdded, OpenType::kKanClosed})) {
                ++kans;
            }
        }

        if (kans < 3) return std::nullopt;
        return 2;
    }

    std::optional<int> YakuEvaluator::HasLittleThreeDragons(const WinningInfo& win_info) noexcept {
        const auto& all_tile_types = win_info.all_tile_types;
        int pons = 0, heads = 0;
        for (const TileType tile_type : {TileType::kWD, TileType::kGD, TileType::kRD}) {
            if (!all_tile_types.count(tile_type)) return std::nullopt;
            if (all_tile_types.at(tile_type) >= 3) ++pons;
            else if (all_tile_types.at(tile_type) == 2) ++heads;
        }
        if (pons == 2 and heads == 1) return 2;
        return std::nullopt;
    }

    bool YakuEvaluator::HasBigThreeDragons(const WinningInfo& win_info) noexcept {
        const auto& all_tile_types = win_info.all_tile_types;
        return all_tile_types.count(TileType::kWD) and all_tile_types.at(TileType::kWD) >= 3 and
               all_tile_types.count(TileType::kGD) and all_tile_types.at(TileType::kGD) >= 3 and
               all_tile_types.count(TileType::kRD) and all_tile_types.at(TileType::kRD) >= 3;
    }

    bool YakuEvaluator::HasAllHonours(const WinningInfo& win_info) noexcept {
        const auto& all_tile_types = win_info.all_tile_types;
        for (const auto& [tile_type, _] : all_tile_types) {
            if (!Is(tile_type, TileSetType::kHonours)) {
                return false;
            }
        }
        return true;
    }

    bool YakuEvaluator::HasAllGreen(const WinningInfo& win_info) noexcept {
        const auto& all_tile_types = win_info.all_tile_types;
        for (const auto& [tile_type, _] : all_tile_types) {
            if (!Is(tile_type, TileSetType::kGreen)) {
                return false;
            }
        }
        return true;
    }

    bool YakuEvaluator::HasAllTerminals(const WinningInfo& win_info) noexcept {
        const auto& all_tile_types = win_info.all_tile_types;
        for (const auto& [tile_type, _] : all_tile_types) {
            if (!Is(tile_type, TileSetType::kTerminals)) {
                return false;
            }
        }
        return true;
    }

    bool YakuEvaluator::HasBigFourWinds(const WinningInfo& win_info) noexcept {
        const auto& all_tile_types = win_info.all_tile_types;
        return all_tile_types.count(TileType::kEW) and all_tile_types.at(TileType::kEW) >= 3 and
               all_tile_types.count(TileType::kSW) and all_tile_types.at(TileType::kSW) >= 3 and
               all_tile_types.count(TileType::kWW) and all_tile_types.at(TileType::kWW) >= 3 and
               all_tile_types.count(TileType::kNW) and all_tile_types.at(TileType::kNW) >= 3;
    }

    bool YakuEvaluator::HasLittleFourWinds(const WinningInfo& win_info) noexcept {
        const auto& all_tile_types = win_info.all_tile_types;
        int pons = 0, heads = 0;
        for (const TileType tile_type : {TileType::kEW, TileType::kSW, TileType::kWW, TileType::kNW}) {
            if (!all_tile_types.count(tile_type)) return false;
            if (all_tile_types.at(tile_type) >= 3) ++pons;
            else if (all_tile_types.at(tile_type) == 2) ++heads;
        }
        return pons == 3 and heads == 1;
    }

    bool YakuEvaluator::HasThirteenOrphans(const WinningInfo& win_info) noexcept {
        const auto& all_tile_types = win_info.all_tile_types;
        assert(win_info.last_added_tile_type);
        const TileType tsumo_type = win_info.last_added_tile_type.value();
        std::map<TileType, int> yaocyu;
        for (const auto& [tile_type, n] : all_tile_types) {
            if (Is(tile_type, TileSetType::kYaocyu)) {
                yaocyu[tile_type] = n;
            }
        }

        return yaocyu.size() == 13 and yaocyu[tsumo_type] == 1;
    }

    bool YakuEvaluator::HasCompletedThirteenOrphans(const WinningInfo& win_info) noexcept {
        const auto& all_tile_types = win_info.all_tile_types;
        assert(win_info.last_added_tile_type);
        const TileType tsumo_type = win_info.last_added_tile_type.value();
        TileTypeCount yaocyu;
        for (const auto& [tile_type, n] : all_tile_types) {
            if (Is(tile_type, TileSetType::kYaocyu)) {
                yaocyu[tile_type] = n;
            }
        }

        return yaocyu.size() == 13 and yaocyu[tsumo_type] == 2;
    }

    bool YakuEvaluator::HasNineGates(const WinningInfo& win_info) noexcept {
        const auto& all_tile_types = win_info.all_tile_types;
        if (!win_info.is_menzen) return false;
        std::map<TileSetType, bool> colors;
        for (const auto& [tile_type, n] : all_tile_types) {
            if (Is(tile_type, TileSetType::kHonours)) return false;
            colors[Color(tile_type)] = true;
        }
        if (colors.size() > 1) return false;

        std::vector<int> required{0,3,1,1,1,1,1,1,1,3};

        assert(win_info.last_added_tile_type);
        const TileType tsumo_type = win_info.last_added_tile_type.value();

        for (const auto& [tile_type, n] : all_tile_types) {
            if (required[Num(tile_type)] > n) return false;
            if (required[Num(tile_type)] < n and tile_type == tsumo_type) return false;
        }

        return true;
    }

    bool YakuEvaluator::HasPureNineGates(const WinningInfo& win_info) noexcept {
        const auto& all_tile_types = win_info.all_tile_types;
        if (!win_info.is_menzen) return false;
        std::map<TileSetType, bool> colors;
        for (const auto& [tile_type, n] : all_tile_types) {
            if (Is(tile_type, TileSetType::kHonours)) return false;
            colors[Color(tile_type)] = true;
        }
        if (colors.size() > 1) return false;

        std::vector<int> required{0,3,1,1,1,1,1,1,1,3};

        assert(win_info.last_added_tile_type);
        const TileType tsumo_type = win_info.last_added_tile_type.value();

        for (const auto& [tile_type, n] : all_tile_types) {
            if (required[Num(tile_type)] > n) return false;
            if (required[Num(tile_type)] == n and tile_type == tsumo_type) return false;
        }

        return true;
    }

    bool YakuEvaluator::HasFourKans(const WinningInfo& win_info) noexcept {
        int kans = 0;
        for (const std::unique_ptr<Open>& open : win_info.opens) {
            if (Any(open->Type(), {
                    OpenType::kKanOpened, OpenType::kKanAdded, OpenType::kKanClosed})) {
                ++kans;
            }
        }

        return kans == 4;
    }

    bool YakuEvaluator::HasFourConcealedPons(const WinningInfo& win_info) noexcept {
        const auto& all_tile_types = win_info.all_tile_types;
        if (!win_info.is_menzen) return false;
        if (all_tile_types.size() != 5) return false;

        assert(win_info.last_added_tile_type);
        const TileType tsumo_type = win_info.last_added_tile_type.value();

        return all_tile_types.at(tsumo_type) > 2;
    }

    bool YakuEvaluator::HasCompletedFourConcealedPons(const WinningInfo& win_info) noexcept {
        const auto& all_tile_types = win_info.all_tile_types;
        if (!win_info.is_menzen) return false;
        if (all_tile_types.size() != 5) return false;

        assert(win_info.last_added_tile_type);
        const TileType tsumo_type = win_info.last_added_tile_type.value();

        return all_tile_types.at(tsumo_type) == 2;
    }
}
