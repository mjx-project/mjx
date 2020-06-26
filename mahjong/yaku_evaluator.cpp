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

        using Sets = std::vector<TileTypeCount>;

        Sets opened_sets;
        for (const Open* open : hand.Opens()) {
            TileTypeCount count;
            for (const Tile tile : open->Tiles()) {
                ++count[tile.Type()];
            }
            opened_sets.push_back(count);
        }

        for (auto [sets, heads] : win_cache_.SetAndHeads(ClosedHandTiles(hand))) {

            for (const TileTypeCount& count : opened_sets) {
                sets.push_back(count);
            }

            std::map<Yaku,int> yaku_in_this_pattern;

            // 各役の判定
            if (std::optional<int> score = HasPinfu(hand, sets, heads); score) {
                yaku_in_this_pattern[Yaku::kPinfu] = score.value();
            }
            if (std::optional<int> score = HasPureDoubleChis(hand, sets, heads); score) {
                yaku_in_this_pattern[Yaku::kPureDoubleChis] = score.value();
            }
            if (std::optional<int> score = HasTwicePureDoubleChis(hand, sets, heads); score) {
                yaku_in_this_pattern[Yaku::kTwicePureDoubleChis] = score.value();
            }
            if (std::optional<int> score = HasSevenPairs(hand, sets, heads); score) {
                yaku_in_this_pattern[Yaku::kSevenPairs] = score.value();
            }
            if (std::optional<int> score = HasAllPons(hand, sets, heads); score) {
                yaku_in_this_pattern[Yaku::kAllPons] = score.value();
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
        for (const auto& [key, val] : yaku) total += val;
        return total;
    }

    std::optional<int> YakuEvaluator::HasFullyConcealdHand(const Hand &hand) const noexcept {
        if (hand.IsMenzen() and hand.Stage() == HandStage::kAfterTsumo) return 1;
        return std::nullopt;
    }

    std::optional<int> YakuEvaluator::HasPinfu(
            const Hand &hand,
            const std::vector<TileTypeCount>& sets,
            const std::vector<TileTypeCount>& heads) const noexcept {
        if (!hand.IsMenzen()) return std::nullopt;

        if (sets.size() != 4 or heads.size() != 1) {
            // 基本形でなければNG.
            return std::nullopt;
        }

        for (const TileTypeCount& count : sets) {
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

        for (const TileTypeCount& st : sets) {
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
            const std::vector<TileTypeCount>& sets,
            const std::vector<TileTypeCount>& heads) const noexcept {

        // 鳴いているとNG
        if (!hand.IsMenzen()) return std::nullopt;

        if (sets.size() != 4 or heads.size() != 1) {
            // 基本形でなければNG.
            return std::nullopt;
        }

        std::map<TileTypeCount, int> pure_chies;
        for (const TileTypeCount& count : sets) {
            if (count.size() == 3) ++pure_chies[count];
        }

        int pairs = 0;
        for (auto& [pure_chie, n] : pure_chies) {
            if (n >= 2) ++pairs;
        }

        // 2つ以上重なっている順子が1個だけあるならOK
        if (pairs == 1) return 1;

        return std::nullopt;
    }

    std::optional<int> YakuEvaluator::HasTwicePureDoubleChis(
            const Hand &hand,
            const std::vector<TileTypeCount>& sets,
            const std::vector<TileTypeCount>& heads) const noexcept {

        // 鳴いているとNG
        if (!hand.IsMenzen()) return std::nullopt;

        if (sets.size() != 4 or heads.size() != 1) {
            // 基本形でなければNG.
            return std::nullopt;
        }

        std::map<TileTypeCount, int> pure_chies;
        for (const TileTypeCount& count : sets) {
            if (count.size() == 3) ++pure_chies[count];
        }

        int pairs = 0;
        for (auto& [pure_chie, n] : pure_chies) {
            if (n >= 2) ++pairs;
        }

        // 2つ以上重なっている順子が2個あるならOK
        if (pairs == 2) return 3;

        return std::nullopt;
    }

    std::optional<int> YakuEvaluator::HasSevenPairs(
            const Hand &hand,
            const std::vector<TileTypeCount>& sets,
            const std::vector<TileTypeCount>& heads) const noexcept {

        if (heads.size() == 7) return 2;
        return std::nullopt;
    }

    std::optional<int> YakuEvaluator::HasAllPons(
            const Hand &hand,
            const std::vector<TileTypeCount>& sets,
            const std::vector<TileTypeCount>& heads) const noexcept {

        if (sets.size() != 4 or heads.size() != 1) {
            // 基本形でなければNG.
            return std::nullopt;
        }

        for (const TileTypeCount& count : sets) {
            if (count.size() >= 3) {
                // 順子が含まれるとNG.
                return std::nullopt;
            }
        }

        return 2;
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