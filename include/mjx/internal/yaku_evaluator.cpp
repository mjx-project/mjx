#include "mjx/internal/yaku_evaluator.h"

#include <cassert>
#include <optional>
#include <tuple>
#include <vector>

#include "mjx/internal/types.h"
#include "mjx/internal/utils.h"
#include "mjx/internal/win_score.h"

namespace mjx::internal {
const WinHandCache& YakuEvaluator::win_cache() {
  return WinHandCache::instance();
}

bool YakuEvaluator::CanWin(const WinInfo& win_info) noexcept {
  WinScore score;

  // closedな手牌が上がり形になっている or 国士無双かどうかを判定する.
  if (!win_cache().Has(win_info.hand.closed_tile_types) and
      !HasThirteenOrphans(win_info) and !HasCompletedThirteenOrphans(win_info))
    return false;

  // 役満の判定
  JudgeYakuman(win_info, score);
  if (!score.yakuman().empty()) return true;

  // 手牌の組み合わせ方によらない役
  JudgeSimpleYaku(win_info, score);
  if (!score.yaku().empty()) return true;

  // 手牌の組み合わせ方に依存する役
  const auto [best_yaku, best_fu, closed_sets, heads] =
      MaximizeTotalFan(win_info);
  for (auto& [yaku, fan] : best_yaku) score.AddYaku(yaku, fan);

  return !score.yaku().empty();
}

WinScore YakuEvaluator::Eval(const WinInfo& win_info) noexcept {
  Assert(win_cache().Has(win_info.hand.closed_tile_types));

  WinScore score;

  // 役満の判定
  JudgeYakuman(win_info, score);
  if (!score.RequireFan()) return score;

  // 手牌の組み合わせ方によらない役
  JudgeSimpleYaku(win_info, score);

  // 手牌の組み合わせ方に依存する役
  const auto [best_yaku, best_fu, closed_sets, heads] =
      MaximizeTotalFan(win_info);
  for (auto& [yaku, fan] : best_yaku) score.AddYaku(yaku, fan);
  score.set_fu(best_fu);

  // 役がないと上がれない.
  Assert(!score.yaku().empty());

  // ドラ
  JudgeDora(win_info, score);

  return score;
}

int YakuEvaluator::CalculateFu(const WinInfo& win_info,
                               const std::vector<TileTypeCount>& closed_sets,
                               const std::vector<TileTypeCount>& heads,
                               const std::map<Yaku, int>& yakus) noexcept {
  // 七対子:25
  if (yakus.count(Yaku::kSevenPairs)) {
    return 25;
  }

  // 平和ツモ:20, 平和ロン:30
  if (yakus.count(Yaku::kPinfu)) {
    if (win_info.hand.stage == HandStage::kAfterTsumo) {
      return 20;
    } else if (win_info.hand.stage == HandStage::kAfterRon) {
      return 30;
    } else {
      Assert(false);
    }
  }

  int fu = 20;

  // 面子
  for (const Open& open : win_info.hand.opens) {
    OpenType open_type = open.Type();
    if (open_type == OpenType::kChi) continue;

    bool is_yaocyu = Is(open.At(0).Type(), TileSetType::kYaocyu);

    switch (open.Type()) {
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
        Assert(false);
    }
  }
  for (const auto& closed_set : closed_sets) {
    if (closed_set.size() > 1) continue;  // 順子は0
    auto type = closed_set.begin()->first;
    auto n = win_info.hand.closed_tile_types.at(type);
    // シャンポンのロンは明刻扱い (n == 4 のときは暗刻扱い #290)
    bool is_ron_triplet = win_info.hand.stage == HandStage::kAfterRon &&
                          type == win_info.hand.win_tile->Type() && n == 3;
    bool is_yaocyu = Is(type, TileSetType::kYaocyu);
    if (is_ron_triplet)
      fu += is_yaocyu ? 4 : 2;  // 明刻
    else
      fu += is_yaocyu ? 8 : 4;  // 暗刻
  }

  // 雀頭
  Assert(heads.size() == 1);
  TileType head_type = heads[0].begin()->first;
  if (Is(head_type, TileSetType::kDragons)) fu += 2;
  // 場風,自風は2符.
  if (Is(head_type, TileSetType::kWinds)) {
    if (IsSameWind(head_type, win_info.state.seat_wind)) fu += 2;
    if (IsSameWind(head_type, win_info.state.prevalent_wind)) fu += 2;
  }

  // 待ち
  bool has_bad_machi = false;
  Assert(win_info.hand.win_tile);
  const TileType tsumo_type = win_info.hand.win_tile.value().Type();
  if (tsumo_type == head_type) has_bad_machi = true;  // 単騎
  for (const TileTypeCount& st : closed_sets) {
    if (st.size() == 1) continue;  // 刻子は弾く
    Assert(st.size() == 3);
    auto it = st.begin();
    const TileType left = it->first;
    ++it;
    const TileType center = it->first;
    ++it;
    const TileType right = it->first;

    if (tsumo_type == center) has_bad_machi = true;  // カンチャン
    if ((tsumo_type == left and Num(right) == 9) or
        (tsumo_type == right and Num(left) == 1)) {
      has_bad_machi = true;  // ペンチャン
    }
  }
  if (has_bad_machi) fu += 2;

  // 面前加符
  auto is_menzen_ron =
      win_info.hand.is_menzen and win_info.hand.stage == HandStage::kAfterRon;
  if (is_menzen_ron) {
    fu += 10;
  }

  // ツモ符
  if (win_info.hand.stage == HandStage::kAfterTsumo) {
    fu += 2;
  }

  // 天鳳は嶺上ツモにも2符加算
  if (win_info.hand.stage == HandStage::kAfterTsumoAfterKan) {
    fu += 2;
  }

  if (fu == 20) {
    // 喰い平和でも最低30符
    return 30;
  }

  // 切り上げ
  if (fu % 10) fu += 10 - fu % 10;

  // 門前ロンはピンフ以外40符以上はあるはず
  Assert(!(win_info.hand.is_menzen &&
           win_info.hand.stage == HandStage::kAfterRon &&
           !yakus.count(Yaku::kPinfu) && fu < 40));
  return fu;
}

std::tuple<std::map<Yaku, int>, int, std::vector<TileTypeCount>,
           std::vector<TileTypeCount>>
YakuEvaluator::MaximizeTotalFan(const WinInfo& win_info) noexcept {
  std::map<Yaku, int> best_yaku;
  int best_fu = 0;
  std::vector<TileTypeCount> best_closed_set, best_heads;

  std::vector<TileTypeCount> opened_sets;
  for (const Open& open : win_info.hand.opens) {
    TileTypeCount count;
    for (const Tile tile : open.Tiles()) {
      ++count[tile.Type()];
    }
    opened_sets.push_back(count);
  }

  for (const auto& [closed_sets, heads] :
       win_cache().SetAndHeads(win_info.hand.closed_tile_types)) {
    std::map<Yaku, int> yaku_in_this_pattern;

    // 各役の判定
    if (const std::optional<int> fan =
            HasPinfu(win_info, closed_sets, opened_sets, heads);
        fan) {
      yaku_in_this_pattern[Yaku::kPinfu] = fan.value();
    }
    if (const std::optional<int> fan =
            HasPureDoubleChis(win_info, closed_sets, opened_sets, heads);
        fan) {
      yaku_in_this_pattern[Yaku::kPureDoubleChis] = fan.value();
    }
    if (const std::optional<int> fan =
            HasTwicePureDoubleChis(win_info, closed_sets, opened_sets, heads);
        fan) {
      yaku_in_this_pattern[Yaku::kTwicePureDoubleChis] = fan.value();
    }
    if (const std::optional<int> fan =
            HasSevenPairs(win_info, closed_sets, opened_sets, heads);
        fan) {
      yaku_in_this_pattern[Yaku::kSevenPairs] = fan.value();
    }
    if (const std::optional<int> fan =
            HasAllPons(win_info, closed_sets, opened_sets, heads);
        fan) {
      yaku_in_this_pattern[Yaku::kAllPons] = fan.value();
    }
    if (const std::optional<int> fan =
            HasPureStraight(win_info, closed_sets, opened_sets, heads);
        fan) {
      yaku_in_this_pattern[Yaku::kPureStraight] = fan.value();
    }
    if (const std::optional<int> fan =
            HasMixedTripleChis(win_info, closed_sets, opened_sets, heads);
        fan) {
      yaku_in_this_pattern[Yaku::kMixedTripleChis] = fan.value();
    }
    if (const std::optional<int> fan =
            HasTriplePons(win_info, closed_sets, opened_sets, heads);
        fan) {
      yaku_in_this_pattern[Yaku::kTriplePons] = fan.value();
    }
    if (const std::optional<int> fan =
            HasOutsideHand(win_info, closed_sets, opened_sets, heads);
        fan) {
      yaku_in_this_pattern[Yaku::kOutsideHand] = fan.value();
    }
    if (const std::optional<int> fan =
            HasTerminalsInAllSets(win_info, closed_sets, opened_sets, heads);
        fan) {
      yaku_in_this_pattern[Yaku::kTerminalsInAllSets] = fan.value();
    }
    if (const std::optional<int> fan =
            HasThreeConcealdPons(win_info, closed_sets, opened_sets, heads);
        fan) {
      yaku_in_this_pattern[Yaku::kThreeConcealedPons] = fan.value();
    }

    // 今までに調べた組み合わせ方より役の総飜数が高いなら採用する
    int best_total_fan = TotalFan(best_yaku);
    int total_fan = TotalFan(yaku_in_this_pattern);
    int fu = CalculateFu(win_info, closed_sets, heads, yaku_in_this_pattern);
    if (best_total_fan < total_fan or
        (best_total_fan == total_fan and best_fu < fu)) {
      std::swap(best_yaku, yaku_in_this_pattern);
      best_closed_set = closed_sets;
      best_heads = heads;
      best_fu = fu;
    }
  }

  return {best_yaku, best_fu, best_closed_set, best_heads};
}

void YakuEvaluator::JudgeYakuman(const WinInfo& win_info,
                                 WinScore& score) noexcept {
  if (HasCompletedThirteenOrphans(win_info)) {
    score.AddYakuman(Yaku::kCompletedThirteenOrphans);
    return;
  }
  if (HasThirteenOrphans(win_info)) {
    score.AddYakuman(Yaku::kThirteenOrphans);
    return;
  }
  if (HasBlessingOfHeaven(win_info)) {
    score.AddYakuman(Yaku::kBlessingOfHeaven);
  }
  if (HasBlessingOfEarth(win_info)) {
    score.AddYakuman(Yaku::kBlessingOfEarth);
  }
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

void YakuEvaluator::JudgeSimpleYaku(const WinInfo& win_info,
                                    WinScore& score) noexcept {
  if (const std::optional<int> fan = HasFullyConcealdHand(win_info); fan) {
    score.AddYaku(Yaku::kFullyConcealedHand, fan.value());
  }
  if (const std::optional<int> fan = HasDoubleRiichi(win_info); fan) {
    score.AddYaku(Yaku::kDoubleRiichi, fan.value());
  } else if (const std::optional<int> fan = HasRiichi(win_info); fan) {
    score.AddYaku(Yaku::kRiichi, fan.value());
  }
  if (const std::optional<int> fan = HasAfterKan(win_info); fan) {
    score.AddYaku(Yaku::kAfterKan, fan.value());
  }
  if (const std::optional<int> fan = HasRobbingKan(win_info); fan) {
    score.AddYaku(Yaku::kRobbingKan, fan.value());
  }
  if (const std::optional<int> fan = HasBottomOfTheSea(win_info); fan) {
    score.AddYaku(Yaku::kBottomOfTheSea, fan.value());
  }
  if (const std::optional<int> fan = HasBottomOfTheRiver(win_info); fan) {
    score.AddYaku(Yaku::kBottomOfTheRiver, fan.value());
  }
  if (const std::optional<int> fan = HasIppatsu(win_info); fan) {
    score.AddYaku(Yaku::kIppatsu, fan.value());
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

void YakuEvaluator::JudgeDora(const WinInfo& win_info,
                              WinScore& score) noexcept {
  if (const std::optional<int> fan = HasRedDora(win_info); fan) {
    score.AddYaku(Yaku::kRedDora, fan.value());
  }
  if (const std::optional<int> fan = HasDora(win_info); fan) {
    score.AddYaku(Yaku::kDora, fan.value());
  }
  if (const std::optional<int> fan = HasReversedDora(win_info); fan) {
    score.AddYaku(Yaku::kReversedDora, fan.value());
  }
}

int YakuEvaluator::TotalFan(const std::map<Yaku, int>& yaku) noexcept {
  int total = 0;
  for (const auto& [key, val] : yaku) total += val;
  return total;
}

std::optional<int> YakuEvaluator::HasRedDora(const WinInfo& win_info) noexcept {
  int reds = 0;
  for (const Tile tile : win_info.hand.closed_tiles) {
    reds += tile.Is(TileSetType::kRedFive);
  }
  for (const Open& open : win_info.hand.opens) {
    for (const Tile tile : open.Tiles()) {
      reds += tile.Is(TileSetType::kRedFive);
    }
  }
  if (reds) return reds;
  return std::nullopt;
}

std::optional<int> YakuEvaluator::HasDora(const WinInfo& win_info) noexcept {
  int dora_count = 0;
  for (const auto& [tile_type, n] : win_info.hand.closed_tile_types) {
    if (win_info.state.dora.count(tile_type)) {
      dora_count += n * win_info.state.dora.at(tile_type);
    }
  }
  for (const Open& open : win_info.hand.opens) {
    for (const Tile tile : open.Tiles()) {
      auto tile_type = tile.Type();
      if (win_info.state.dora.count(tile_type)) {
        dora_count += win_info.state.dora.at(tile_type);
      }
    }
  }
  if (dora_count > 0) return dora_count;
  return std::nullopt;
}

std::optional<int> YakuEvaluator::HasReversedDora(
    const WinInfo& win_info) noexcept {
  if (!win_info.hand.under_riichi) return std::nullopt;

  int dora_count = 0;
  for (const auto& [tile_type, n] : win_info.hand.closed_tile_types) {
    if (win_info.state.reversed_dora.count(tile_type)) {
      dora_count += n * win_info.state.reversed_dora.at(tile_type);
    }
  }
  for (const Open& open : win_info.hand.opens) {
    for (const Tile tile : open.Tiles()) {
      auto tile_type = tile.Type();
      if (win_info.state.reversed_dora.count(tile_type)) {
        dora_count += win_info.state.reversed_dora.at(tile_type);
      }
    }
  }
  return dora_count;
}

std::optional<int> YakuEvaluator::HasPinfu(
    const WinInfo& win_info, const std::vector<TileTypeCount>& closed_sets,
    const std::vector<TileTypeCount>& opened_sets,
    const std::vector<TileTypeCount>& heads) noexcept {
  if (!win_info.hand.is_menzen) return std::nullopt;

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

  // 雀頭が役牌・自風・場風だとNG
  const TileType head = heads[0].begin()->first;
  if (Is(head, TileSetType::kDragons)) return std::nullopt;
  if (IsSameWind(head, win_info.state.seat_wind)) return std::nullopt;
  if (IsSameWind(head, win_info.state.prevalent_wind)) return std::nullopt;

  Assert(win_info.hand.win_tile);
  const TileType tsumo_type = win_info.hand.win_tile.value().Type();

  for (const TileTypeCount& st : closed_sets) {
    const TileType left = st.begin()->first, right = st.rbegin()->first;
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
    const WinInfo& win_info, const std::vector<TileTypeCount>& closed_sets,
    const std::vector<TileTypeCount>& opened_sets,
    const std::vector<TileTypeCount>& heads) noexcept {
  // 鳴いているとNG
  if (!win_info.hand.is_menzen) return std::nullopt;

  // 基本形でなければNG。暗槓もOK
  if (closed_sets.size() + opened_sets.size() != 4 or heads.size() != 1) {
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
    const WinInfo& win_info, const std::vector<TileTypeCount>& closed_sets,
    const std::vector<TileTypeCount>& opened_sets,
    const std::vector<TileTypeCount>& heads) noexcept {
  // 鳴いているとNG
  if (!win_info.hand.is_menzen) return std::nullopt;

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
    const WinInfo& win_info, const std::vector<TileTypeCount>& closed_sets,
    const std::vector<TileTypeCount>& opened_sets,
    const std::vector<TileTypeCount>& heads) noexcept {
  if (heads.size() == 7) return 2;
  return std::nullopt;
}

std::optional<int> YakuEvaluator::HasAllPons(
    const WinInfo& win_info, const std::vector<TileTypeCount>& closed_sets,
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
    const WinInfo& win_info, const std::vector<TileTypeCount>& closed_sets,
    const std::vector<TileTypeCount>& opened_sets,
    const std::vector<TileTypeCount>& heads) noexcept {
  if (opened_sets.size() + closed_sets.size() != 4 or heads.size() != 1) {
    // 基本形でなければNG.
    return std::nullopt;
  }

  std::map<TileTypeCount, bool> chis;
  for (const std::vector<TileTypeCount>& sets : {closed_sets, opened_sets}) {
    for (const TileTypeCount& count : sets) {
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
    if (win_info.hand.is_menzen)
      return 2;
    else
      return 1;
  }

  return std::nullopt;
}

std::optional<int> YakuEvaluator::HasMixedTripleChis(
    const WinInfo& win_info, const std::vector<TileTypeCount>& closed_sets,
    const std::vector<TileTypeCount>& opened_sets,
    const std::vector<TileTypeCount>& heads) noexcept {
  if (opened_sets.size() + closed_sets.size() != 4 or heads.size() != 1) {
    // 基本形でなければNG.
    return std::nullopt;
  }

  std::map<TileTypeCount, bool> chis;
  for (const std::vector<TileTypeCount>& sets : {closed_sets, opened_sets}) {
    for (const TileTypeCount& count : sets) {
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
    if (win_info.hand.is_menzen)
      return 2;
    else
      return 1;
  }

  return std::nullopt;
}

std::optional<int> YakuEvaluator::HasTriplePons(
    const WinInfo& win_info, const std::vector<TileTypeCount>& closed_sets,
    const std::vector<TileTypeCount>& opened_sets,
    const std::vector<TileTypeCount>& heads) noexcept {
  if (opened_sets.size() + closed_sets.size() != 4 or heads.size() != 1) {
    // 基本形でなければNG.
    return std::nullopt;
  }

  std::map<TileTypeCount, bool> pons;
  for (const std::vector<TileTypeCount>& sets : {closed_sets, opened_sets}) {
    for (const TileTypeCount& count : sets) {
      if (count.size() == 1) pons[count] = true;
    }
  }

  bool has_triple_pons = false;

  for (int start = 0; start < 9; ++start) {
    bool valid = true;
    for (int set_start : {start, start + 9, start + 18}) {
      TileTypeCount count;
      bool has_kotsu = false;
      // ３個の刻子をもつ
      count[static_cast<TileType>(set_start)] = 3;
      has_kotsu |= pons.count(count);
      // ４個の刻子（槓子）をもつ
      count[static_cast<TileType>(set_start)] = 4;
      has_kotsu |= pons.count(count);
      if (!has_kotsu) {
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
    const WinInfo& win_info, const std::vector<TileTypeCount>& closed_sets,
    const std::vector<TileTypeCount>& opened_sets,
    const std::vector<TileTypeCount>& heads) noexcept {
  bool has_honour = false;
  bool all_yaochu = true;

  for (const std::vector<TileTypeCount>& blocks :
       {closed_sets, opened_sets, heads}) {
    for (const TileTypeCount& count : blocks) {
      bool valid = false;
      for (auto& [tile_type, n] : count) {
        if (!has_honour and Is(tile_type, TileSetType::kHonours)) {
          has_honour = true;
        }
        if (Is(tile_type, TileSetType::kYaocyu)) {
          valid = true;
          if (n >= 2) break;
        } else {
          all_yaochu = false;
        }
      }
      if (!valid) return std::nullopt;
    }
  }

  // 純全帯幺とは複合しない
  if (!has_honour) return std::nullopt;

  // 混老頭とは複合しない
  if (all_yaochu) return std::nullopt;

  if (win_info.hand.is_menzen)
    return 2;
  else
    return 1;
}

std::optional<int> YakuEvaluator::HasTerminalsInAllSets(
    const WinInfo& win_info, const std::vector<TileTypeCount>& closed_sets,
    const std::vector<TileTypeCount>& opened_sets,
    const std::vector<TileTypeCount>& heads) noexcept {
  for (const std::vector<TileTypeCount>& blocks :
       {closed_sets, opened_sets, heads}) {
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

  if (win_info.hand.is_menzen)
    return 3;
  else
    return 2;
}

std::optional<int> YakuEvaluator::HasFullyConcealdHand(
    const WinInfo& win_info) noexcept {
  if (win_info.hand.is_menzen and
      Any(win_info.hand.stage,
          {HandStage::kAfterTsumo, HandStage::kAfterTsumoAfterKan}))
    return 1;
  return std::nullopt;
}

std::optional<int> YakuEvaluator::HasRiichi(const WinInfo& win_info) noexcept {
  if (win_info.hand.under_riichi) return 1;
  return std::nullopt;
}
std::optional<int> YakuEvaluator::HasDoubleRiichi(
    const WinInfo& win_info) noexcept {
  if (win_info.hand.double_riichi) return 2;
  return std::nullopt;
}

std::optional<int> YakuEvaluator::HasAfterKan(
    const WinInfo& win_info) noexcept {
  if (win_info.hand.stage == HandStage::kAfterTsumoAfterKan) return 1;
  return std::nullopt;
}

std::optional<int> YakuEvaluator::HasRobbingKan(
    const WinInfo& win_info) noexcept {
  if (win_info.state.is_robbing_kan) return 1;
  return std::nullopt;
}

std::optional<int> YakuEvaluator::HasBottomOfTheSea(
    const WinInfo& win_info) noexcept {
  if (win_info.state.is_bottom and
      win_info.hand.stage == HandStage::kAfterTsumo)
    return 1;
  return std::nullopt;
}

std::optional<int> YakuEvaluator::HasBottomOfTheRiver(
    const WinInfo& win_info) noexcept {
  if (win_info.state.is_bottom and win_info.hand.stage == HandStage::kAfterRon)
    return 1;
  return std::nullopt;
}

std::optional<int> YakuEvaluator::HasIppatsu(const WinInfo& win_info) noexcept {
  if (win_info.state.is_ippatsu) return 1;
  return std::nullopt;
}

std::optional<int> YakuEvaluator::HasAllSimples(
    const WinInfo& win_info) noexcept {
  for (const auto& [tile_type, _] : win_info.hand.all_tile_types) {
    if (Is(tile_type, TileSetType::kYaocyu)) return std::nullopt;
  }
  return 1;
}

std::optional<int> YakuEvaluator::HasWhiteDragon(
    const WinInfo& win_info) noexcept {
  const auto& all_tile_types = win_info.hand.all_tile_types;
  if (all_tile_types.count(TileType::kWD) and
      all_tile_types.at(TileType::kWD) >= 3)
    return 1;
  return std::nullopt;
}

std::optional<int> YakuEvaluator::HasGreenDragon(
    const WinInfo& win_info) noexcept {
  const auto& all_tile_types = win_info.hand.all_tile_types;
  if (all_tile_types.count(TileType::kGD) and
      all_tile_types.at(TileType::kGD) >= 3)
    return 1;
  return std::nullopt;
}

std::optional<int> YakuEvaluator::HasRedDragon(
    const WinInfo& win_info) noexcept {
  const auto& all_tile_types = win_info.hand.all_tile_types;
  if (all_tile_types.count(TileType::kRD) and
      all_tile_types.at(TileType::kRD) >= 3)
    return 1;
  return std::nullopt;
}

std::optional<int> YakuEvaluator::HasSeatWindEast(
    const WinInfo& win_info) noexcept {
  if (win_info.state.seat_wind != Wind::kEast) return std::nullopt;
  const auto& all_tile_types = win_info.hand.all_tile_types;
  if (all_tile_types.count(TileType::kEW) and
      all_tile_types.at(TileType::kEW) >= 3)
    return 1;
  return std::nullopt;
}

std::optional<int> YakuEvaluator::HasSeatWindSouth(
    const WinInfo& win_info) noexcept {
  if (win_info.state.seat_wind != Wind::kSouth) return std::nullopt;
  const auto& all_tile_types = win_info.hand.all_tile_types;
  if (all_tile_types.count(TileType::kSW) and
      all_tile_types.at(TileType::kSW) >= 3)
    return 1;
  return std::nullopt;
}

std::optional<int> YakuEvaluator::HasSeatWindWest(
    const WinInfo& win_info) noexcept {
  if (win_info.state.seat_wind != Wind::kWest) return std::nullopt;
  const auto& all_tile_types = win_info.hand.all_tile_types;
  if (all_tile_types.count(TileType::kWW) and
      all_tile_types.at(TileType::kWW) >= 3)
    return 1;
  return std::nullopt;
}

std::optional<int> YakuEvaluator::HasSeatWindNorth(
    const WinInfo& win_info) noexcept {
  if (win_info.state.seat_wind != Wind::kNorth) return std::nullopt;
  const auto& all_tile_types = win_info.hand.all_tile_types;
  if (all_tile_types.count(TileType::kNW) and
      all_tile_types.at(TileType::kNW) >= 3)
    return 1;
  return std::nullopt;
}

std::optional<int> YakuEvaluator::HasPrevalentWindEast(
    const WinInfo& win_info) noexcept {
  if (win_info.state.prevalent_wind != Wind::kEast) return std::nullopt;
  const auto& all_tile_types = win_info.hand.all_tile_types;
  if (all_tile_types.count(TileType::kEW) and
      all_tile_types.at(TileType::kEW) >= 3)
    return 1;
  return std::nullopt;
}

std::optional<int> YakuEvaluator::HasPrevalentWindSouth(
    const WinInfo& win_info) noexcept {
  if (win_info.state.prevalent_wind != Wind::kSouth) return std::nullopt;
  const auto& all_tile_types = win_info.hand.all_tile_types;
  if (all_tile_types.count(TileType::kSW) and
      all_tile_types.at(TileType::kSW) >= 3)
    return 1;
  return std::nullopt;
}

std::optional<int> YakuEvaluator::HasPrevalentWindWest(
    const WinInfo& win_info) noexcept {
  if (win_info.state.prevalent_wind != Wind::kWest) return std::nullopt;
  const auto& all_tile_types = win_info.hand.all_tile_types;
  if (all_tile_types.count(TileType::kWW) and
      all_tile_types.at(TileType::kWW) >= 3)
    return 1;
  return std::nullopt;
}

std::optional<int> YakuEvaluator::HasPrevalentWindNorth(
    const WinInfo& win_info) noexcept {
  if (win_info.state.prevalent_wind != Wind::kNorth) return std::nullopt;
  const auto& all_tile_types = win_info.hand.all_tile_types;
  if (all_tile_types.count(TileType::kNW) and
      all_tile_types.at(TileType::kNW) >= 3)
    return 1;
  return std::nullopt;
}

std::optional<int> YakuEvaluator::HasAllTermsAndHonours(
    const WinInfo& win_info) noexcept {
  const auto& all_tile_types = win_info.hand.all_tile_types;
  for (const auto& [tile_type, _] : all_tile_types) {
    if (Is(tile_type, TileSetType::kTanyao)) return std::nullopt;
  }
  return 2;
}

std::optional<int> YakuEvaluator::HasHalfFlush(
    const WinInfo& win_info) noexcept {
  const auto& all_tile_types = win_info.hand.all_tile_types;
  std::map<TileSetType, bool> set_types;
  for (const auto& [tile_type, _] : all_tile_types) {
    if (Is(tile_type, TileSetType::kHonours))
      set_types[TileSetType::kHonours] = true;
    else
      set_types[Color(tile_type)] = true;
  }

  if (set_types.count(TileSetType::kHonours) == 0 or set_types.size() > 2)
    return std::nullopt;
  if (win_info.hand.is_menzen) return 3;
  return 2;
}

std::optional<int> YakuEvaluator::HasFullFlush(
    const WinInfo& win_info) noexcept {
  const auto& all_tile_types = win_info.hand.all_tile_types;
  std::map<TileSetType, bool> set_types;
  for (const auto& [tile_type, _] : all_tile_types) {
    if (Is(tile_type, TileSetType::kHonours))
      set_types[TileSetType::kHonours] = true;
    else
      set_types[Color(tile_type)] = true;
  }

  if (set_types.count(TileSetType::kHonours) or set_types.size() > 1)
    return std::nullopt;
  if (win_info.hand.is_menzen) return 6;
  return 5;
}

std::optional<int> YakuEvaluator::HasThreeKans(
    const WinInfo& win_info) noexcept {
  int kans = 0;
  for (const Open& open : win_info.hand.opens) {
    if (Any(open.Type(), {OpenType::kKanOpened, OpenType::kKanAdded,
                          OpenType::kKanClosed})) {
      ++kans;
    }
  }

  if (kans < 3) return std::nullopt;
  return 2;
}

std::optional<int> YakuEvaluator::HasLittleThreeDragons(
    const WinInfo& win_info) noexcept {
  const auto& all_tile_types = win_info.hand.all_tile_types;
  int pons = 0, heads = 0;
  for (const TileType tile_type :
       {TileType::kWD, TileType::kGD, TileType::kRD}) {
    if (!all_tile_types.count(tile_type)) return std::nullopt;
    if (all_tile_types.at(tile_type) >= 3)
      ++pons;
    else if (all_tile_types.at(tile_type) == 2)
      ++heads;
  }
  if (pons == 2 and heads == 1) return 2;
  return std::nullopt;
}

bool YakuEvaluator::HasBlessingOfHeaven(const WinInfo& win_info) noexcept {
  return win_info.state.is_first_tsumo and win_info.state.is_dealer;
}

bool YakuEvaluator::HasBlessingOfEarth(const WinInfo& win_info) noexcept {
  return win_info.state.is_first_tsumo and !win_info.state.is_dealer;
}

bool YakuEvaluator::HasBigThreeDragons(const WinInfo& win_info) noexcept {
  const auto& all_tile_types = win_info.hand.all_tile_types;
  return all_tile_types.count(TileType::kWD) and
         all_tile_types.at(TileType::kWD) >= 3 and
         all_tile_types.count(TileType::kGD) and
         all_tile_types.at(TileType::kGD) >= 3 and
         all_tile_types.count(TileType::kRD) and
         all_tile_types.at(TileType::kRD) >= 3;
}

bool YakuEvaluator::HasAllHonours(const WinInfo& win_info) noexcept {
  const auto& all_tile_types = win_info.hand.all_tile_types;
  for (const auto& [tile_type, _] : all_tile_types) {
    if (!Is(tile_type, TileSetType::kHonours)) {
      return false;
    }
  }
  return true;
}

bool YakuEvaluator::HasAllGreen(const WinInfo& win_info) noexcept {
  const auto& all_tile_types = win_info.hand.all_tile_types;
  for (const auto& [tile_type, _] : all_tile_types) {
    if (!Is(tile_type, TileSetType::kGreen)) {
      return false;
    }
  }
  return true;
}

bool YakuEvaluator::HasAllTerminals(const WinInfo& win_info) noexcept {
  const auto& all_tile_types = win_info.hand.all_tile_types;
  for (const auto& [tile_type, _] : all_tile_types) {
    if (!Is(tile_type, TileSetType::kTerminals)) {
      return false;
    }
  }
  return true;
}

bool YakuEvaluator::HasBigFourWinds(const WinInfo& win_info) noexcept {
  const auto& all_tile_types = win_info.hand.all_tile_types;
  return all_tile_types.count(TileType::kEW) and
         all_tile_types.at(TileType::kEW) >= 3 and
         all_tile_types.count(TileType::kSW) and
         all_tile_types.at(TileType::kSW) >= 3 and
         all_tile_types.count(TileType::kWW) and
         all_tile_types.at(TileType::kWW) >= 3 and
         all_tile_types.count(TileType::kNW) and
         all_tile_types.at(TileType::kNW) >= 3;
}

bool YakuEvaluator::HasLittleFourWinds(const WinInfo& win_info) noexcept {
  const auto& all_tile_types = win_info.hand.all_tile_types;
  int pons = 0, heads = 0;
  for (const TileType tile_type :
       {TileType::kEW, TileType::kSW, TileType::kWW, TileType::kNW}) {
    if (!all_tile_types.count(tile_type)) return false;
    if (all_tile_types.at(tile_type) >= 3)
      ++pons;
    else if (all_tile_types.at(tile_type) == 2)
      ++heads;
  }
  return pons == 3 and heads == 1;
}

bool YakuEvaluator::HasThirteenOrphans(const WinInfo& win_info) noexcept {
  const auto& all_tile_types = win_info.hand.all_tile_types;
  Assert(win_info.hand.win_tile);
  const TileType tsumo_type = win_info.hand.win_tile.value().Type();
  std::map<TileType, int> yaocyu;
  for (const auto& [tile_type, n] : all_tile_types) {
    if (Is(tile_type, TileSetType::kYaocyu)) {
      yaocyu[tile_type] = n;
    }
  }

  return yaocyu.size() == 13 and yaocyu[tsumo_type] == 1;
}

bool YakuEvaluator::HasCompletedThirteenOrphans(
    const WinInfo& win_info) noexcept {
  const auto& all_tile_types = win_info.hand.all_tile_types;
  Assert(win_info.hand.win_tile);
  const TileType tsumo_type = win_info.hand.win_tile.value().Type();
  TileTypeCount yaocyu;
  for (const auto& [tile_type, n] : all_tile_types) {
    if (Is(tile_type, TileSetType::kYaocyu)) {
      yaocyu[tile_type] = n;
    }
  }

  return yaocyu.size() == 13 and yaocyu[tsumo_type] == 2;
}

bool YakuEvaluator::HasNineGates(const WinInfo& win_info) noexcept {
  const auto& all_tile_types = win_info.hand.all_tile_types;
  if (!win_info.hand.is_menzen) return false;
  std::map<TileSetType, bool> colors;
  for (const auto& [tile_type, n] : all_tile_types) {
    if (Is(tile_type, TileSetType::kHonours)) return false;
    colors[Color(tile_type)] = true;
  }
  if (colors.size() > 1) return false;
  if (all_tile_types.size() < 9) return false;

  std::vector<int> required{0, 3, 1, 1, 1, 1, 1, 1, 1, 3};

  Assert(win_info.hand.win_tile);
  const TileType tsumo_type = win_info.hand.win_tile.value().Type();

  for (const auto& [tile_type, n] : all_tile_types) {
    if (required[Num(tile_type)] > n) return false;
    if (required[Num(tile_type)] < n and tile_type == tsumo_type) return false;
  }

  return true;
}

bool YakuEvaluator::HasPureNineGates(const WinInfo& win_info) noexcept {
  const auto& all_tile_types = win_info.hand.all_tile_types;
  if (!win_info.hand.is_menzen) return false;
  std::map<TileSetType, bool> colors;
  for (const auto& [tile_type, n] : all_tile_types) {
    if (Is(tile_type, TileSetType::kHonours)) return false;
    colors[Color(tile_type)] = true;
  }
  if (colors.size() > 1) return false;
  if (all_tile_types.size() < 9) return false;

  std::vector<int> required{0, 3, 1, 1, 1, 1, 1, 1, 1, 3};

  Assert(win_info.hand.win_tile);
  const TileType tsumo_type = win_info.hand.win_tile.value().Type();

  for (const auto& [tile_type, n] : all_tile_types) {
    if (required[Num(tile_type)] > n) return false;
    if (required[Num(tile_type)] == n and tile_type == tsumo_type) return false;
  }

  return true;
}

bool YakuEvaluator::HasFourKans(const WinInfo& win_info) noexcept {
  int kans = 0;
  for (const Open& open : win_info.hand.opens) {
    if (Any(open.Type(), {OpenType::kKanOpened, OpenType::kKanAdded,
                          OpenType::kKanClosed})) {
      ++kans;
    }
  }

  return kans == 4;
}

bool YakuEvaluator::HasFourConcealedPons(const WinInfo& win_info) noexcept {
  const auto& all_tile_types = win_info.hand.all_tile_types;
  if (!win_info.hand.is_menzen) return false;
  if (win_info.hand.stage == HandStage::kAfterRon)
    return false;  // ロンのときは四暗刻単騎のみ
  if (all_tile_types.size() != 5) return false;

  // ２個以下の要素は１つだけ
  int count = 0;
  for (const auto& [type, cnt] : all_tile_types) {
    count += cnt <= 2;
  }
  if (count > 1) return false;

  Assert(win_info.hand.win_tile);
  const TileType tsumo_type = win_info.hand.win_tile.value().Type();

  return all_tile_types.at(tsumo_type) > 2;
}

bool YakuEvaluator::HasCompletedFourConcealedPons(
    const WinInfo& win_info) noexcept {
  const auto& all_tile_types = win_info.hand.all_tile_types;
  if (!win_info.hand.is_menzen) return false;
  if (all_tile_types.size() != 5) return false;

  // ２個以下の要素は１つだけ
  int count = 0;
  for (const auto& [type, cnt] : all_tile_types) {
    count += cnt <= 2;
  }
  if (count > 1) return false;

  Assert(win_info.hand.win_tile);
  const TileType tsumo_type = win_info.hand.win_tile.value().Type();

  return all_tile_types.at(tsumo_type) == 2;
}

std::optional<int> YakuEvaluator::HasThreeConcealdPons(
    const WinInfo& win_info, const std::vector<TileTypeCount>& closed_sets,
    const std::vector<TileTypeCount>& opened_sets,
    const std::vector<TileTypeCount>& heads) noexcept {
  int cnt_triplets = 0;
  // 暗槓
  for (const auto& open : win_info.hand.opens)
    if (open.Type() == OpenType::kKanClosed) ++cnt_triplets;
  // 暗刻
  for (const TileTypeCount& count : closed_sets) {
    if (count.size() != 1) continue;
    bool is_triplet = count.begin()->second == 3;
    // 刻子でもロンだと明刻扱い
    bool is_not_ron = win_info.hand.stage != HandStage::kAfterRon ||
                      win_info.hand.win_tile->Type() != count.begin()->first;
    // 4枚目ならロンであろうが暗刻 PR#311
    bool is_quad =
        win_info.hand.closed_tile_types.at(count.begin()->first) == 4;
    if (is_triplet && (is_not_ron || is_quad)) ++cnt_triplets;
  }
  return cnt_triplets >= 3 ? std::make_optional(2) : std::nullopt;
}
}  // namespace mjx::internal
