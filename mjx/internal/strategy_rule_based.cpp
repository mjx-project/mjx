#include "mjx/internal/strategy_rule_based.h"

#include "mjx/internal/utils.h"

namespace mjx::internal {
std::vector<mjxproto::Action> StrategyRuleBased::TakeActions(
    std::vector<Observation> &&observations) const {
  int N = observations.size();
  std::vector<mjxproto::Action> actions(N);
  for (int i = 0; i < N; ++i) {
    actions[i] = TakeAction(std::move(observations[i]));
  }
  return actions;
}

mjxproto::Action StrategyRuleBased::TakeAction(
    Observation &&observation) const {
  // Prepare some seed and MT engine for reproducibility
  const std::uint64_t seed =
      12345 + 4096 * observation.proto().public_observation().events_size() +
      16 * observation.legal_actions().size() + 1 * observation.proto().who();
  auto mt = std::mt19937_64(seed);

  auto legal_actions = observation.legal_actions();

  // もし、取りうる行動が一種類ならそれをそのまま返す
  if (legal_actions.size() == 1) return legal_actions[0];

  // この順番でソート
  std::unordered_map<mjxproto::ActionType, int> action_priority = {
      {mjxproto::ACTION_TYPE_TSUMO, 0},
      {mjxproto::ACTION_TYPE_RIICHI, 1},
      {mjxproto::ACTION_TYPE_ABORTIVE_DRAW_NINE_TERMINALS, 2},
      {mjxproto::ACTION_TYPE_CLOSED_KAN, 3},
      {mjxproto::ACTION_TYPE_ADDED_KAN, 4},
      {mjxproto::ACTION_TYPE_TSUMOGIRI, 5},
      {mjxproto::ACTION_TYPE_DISCARD, 6},
      {mjxproto::ACTION_TYPE_RON, 7},
      {mjxproto::ACTION_TYPE_PON, 8},
      {mjxproto::ACTION_TYPE_OPEN_KAN, 9},
      {mjxproto::ACTION_TYPE_CHI, 10},
      {mjxproto::ACTION_TYPE_NO, 11},
  };
  std::sort(
      legal_actions.begin(), legal_actions.end(),
      [&action_priority](const mjxproto::Action &x, const mjxproto::Action &y) {
        if (x.type() != y.type())
          return action_priority.at(x.type()) < action_priority.at(y.type());
        else
          return x.open() < y.open();
      });

  const Hand curr_hand = observation.current_hand();

  auto &selected = legal_actions.front();

  // 和了れるときは全て和了る。リーチできるときは全てリーチする。九種九牌も全て流す。
  if (Any(selected.type(),
          {mjxproto::ACTION_TYPE_TSUMO, mjxproto::ACTION_TYPE_RIICHI,
           mjxproto::ACTION_TYPE_RON,
           mjxproto::ACTION_TYPE_ABORTIVE_DRAW_NINE_TERMINALS})) {
    return selected;
  }

  // テンパっているときには他家から鳴かない
  if (Any(selected.type(),
          {mjxproto::ACTION_TYPE_OPEN_KAN, mjxproto::ACTION_TYPE_PON,
           mjxproto::ACTION_TYPE_CHI})) {
    if (curr_hand.IsTenpai()) {
      selected = *legal_actions.rbegin();
      Assert(selected.type() == mjxproto::ActionType::ACTION_TYPE_NO);
      return selected;
    }
  }

  // 鳴ける場合にはランダムに行動選択
  if (Any(selected.type(),
          {mjxproto::ACTION_TYPE_OPEN_KAN, mjxproto::ACTION_TYPE_PON,
           mjxproto::ACTION_TYPE_CHI})) {
    selected = *SelectRandomly(legal_actions.begin(), legal_actions.end(), mt);
    Assert(Any(selected.type(),
               {mjxproto::ACTION_TYPE_OPEN_KAN, mjxproto::ACTION_TYPE_PON,
                mjxproto::ACTION_TYPE_CHI, mjxproto::ACTION_TYPE_NO}));
    return selected;
  }

  // 暗槓、加槓ができるときはランダムに実行
  if (Any(selected.type(), {mjxproto::ACTION_TYPE_CLOSED_KAN,
                            mjxproto::ACTION_TYPE_ADDED_KAN})) {
    auto prob = RandomProb(mt);
    if (prob > 0.5) return selected;
  }

  // Discardが選択されたとき（あるいはdiscardしかできないとき）、切る牌を選ぶ
  std::vector<std::pair<Tile, bool>> possible_discards =
      observation.possible_discards();
  std::vector<Tile> discard_candidates;
  for (const auto &[tile, tsumogiri] : possible_discards)
    discard_candidates.emplace_back(tile);
  Tile selected_discard = SelectDiscard(discard_candidates, curr_hand, mt);
  for (const auto &legal_action : legal_actions) {
    if (!Any(legal_action.type(),
             {mjxproto::ACTION_TYPE_DISCARD, mjxproto::ACTION_TYPE_TSUMOGIRI}))
      continue;
    if (legal_action.tile() == selected_discard.Id()) return legal_action;
  }
  assert(false);
}

template <typename RandomGenerator>
Tile StrategyRuleBased::SelectDiscard(std::vector<Tile> &discard_candidates,
                                      const Hand &curr_hand,
                                      RandomGenerator &g) {
  std::sort(discard_candidates.begin(), discard_candidates.end());
  const TileTypeCount closed_tile_type_cnt = curr_hand.ClosedTileTypes();
  // 聴牌が取れるなら取れるように切る
  if (curr_hand.CanTakeTenpai()) {
    auto tenpai_discards = curr_hand.PossibleDiscardsToTakeTenpai();
    for (const auto tile : discard_candidates) {
      if (std::any_of(tenpai_discards.begin(), tenpai_discards.end(),
                      [&tile](const auto &x) { return x.first == tile; })) {
        return tile;
      }
    }
    Assert(false, "discard_candidates: " + Tile::ToString(discard_candidates) +
                      "\ncurr_hand.ToVectorClosed(): " +
                      Tile::ToString(curr_hand.ToVectorClosed(true)));
  }
  // 判定ロジック
  auto is_head = [&closed_tile_type_cnt](Tile tile) {
    return closed_tile_type_cnt.count(tile.Type()) &&
           closed_tile_type_cnt.at(tile.Type()) >= 2;
  };
  auto is_pon = [&closed_tile_type_cnt](Tile tile) {
    return closed_tile_type_cnt.count(tile.Type()) &&
           closed_tile_type_cnt.at(tile.Type()) >= 3;
  };
  auto has_next = [&closed_tile_type_cnt](Tile tile, int n) {
    return (bool)closed_tile_type_cnt.count(TileType(ToUType(tile.Type()) + n));
  };
  auto has_prev = [&closed_tile_type_cnt](Tile tile, int n) {
    return (bool)closed_tile_type_cnt.count(TileType(ToUType(tile.Type()) - n));
  };
  auto is_chi = [&](Tile tile) {
    if (Any(tile.Type(), {TileType::kM1, TileType::kP1, TileType::kS1}))
      return has_next(tile, 1) && has_next(tile, 2);
    if (Any(tile.Type(), {TileType::kM2, TileType::kP2, TileType::kS2}))
      return (has_next(tile, 1) && has_next(tile, 2)) ||
             (has_prev(tile, 1) && has_next(tile, 1));
    if (Any(tile.Type(), {TileType::kM9, TileType::kP9, TileType::kS9}))
      return has_prev(tile, 1) && has_prev(tile, 2);
    if (Any(tile.Type(), {TileType::kM8, TileType::kP8, TileType::kS8}))
      return (has_prev(tile, 1) && has_prev(tile, 2)) ||
             (has_prev(tile, 1) && has_next(tile, 1));
    return (has_next(tile, 1) && has_next(tile, 2)) ||
           (has_prev(tile, 1) && has_prev(tile, 2)) ||
           (has_prev(tile, 1) && has_next(tile, 1));
  };
  auto has_neighbors = [&](Tile tile) {
    if (Any(tile.Type(), {TileType::kM1, TileType::kP1, TileType::kS1}))
      return has_next(tile, 1);
    if (Any(tile.Type(), {TileType::kM9, TileType::kP9, TileType::kS9}))
      return has_prev(tile, 1);
    return has_next(tile, 1) || has_prev(tile, 1);
  };
  auto has_skip_neighbors = [&](Tile tile) {
    if (Any(tile.Type(), {TileType::kM1, TileType::kP1, TileType::kS1,
                          TileType::kM2, TileType::kP2, TileType::kS2}))
      return has_next(tile, 2);
    if (Any(tile.Type(), {TileType::kM9, TileType::kP9, TileType::kS9,
                          TileType::kM8, TileType::kP8, TileType::kS8}))
      return has_prev(tile, 2);
    return has_next(tile, 2) || has_prev(tile, 2);
  };
  // 字牌孤立牌があればまずそれを切り飛ばす
  for (const auto tile : discard_candidates) {
    if (!Is(tile.Type(), TileSetType::kHonours)) continue;
    if (is_head(tile) || is_pon(tile)) continue;
    return tile;
  }
  // 19の孤立牌を切り飛ばす
  for (const auto tile : discard_candidates) {
    if (!Is(tile.Type(), TileSetType::kTerminals)) continue;
    if (is_head(tile) || is_pon(tile) || is_chi(tile) || has_neighbors(tile) ||
        has_skip_neighbors(tile))
      continue;
    return tile;
  }
  // 断么九の孤立牌を切り飛ばす
  for (const auto tile : discard_candidates) {
    if (!Is(tile.Type(), TileSetType::kTanyao)) continue;
    if (is_head(tile) || is_pon(tile) || is_chi(tile) || has_neighbors(tile) ||
        has_skip_neighbors(tile))
      continue;
    return tile;
  }
  // 19ペンチャンを外す
  for (const auto tile : discard_candidates) {
    if (!Is(tile.Type(), TileSetType::kTerminals)) continue;
    if (is_head(tile) || is_pon(tile) || is_chi(tile) ||
        has_skip_neighbors(tile))
      continue;
    return tile;
  }
  // 19カンチャンを外す
  for (const auto tile : discard_candidates) {
    if (!Is(tile.Type(), TileSetType::kTerminals)) continue;
    if (is_head(tile) || is_pon(tile) || is_chi(tile) || has_neighbors(tile))
      continue;
    return tile;
  }
  // 断么九のカンチャンを外す
  for (const auto tile : discard_candidates) {
    if (!Is(tile.Type(), TileSetType::kTanyao)) continue;
    if (is_head(tile) || is_pon(tile) || is_chi(tile) || has_neighbors(tile))
      continue;
    return tile;
  }
  // 断么九の両面を外す
  for (const auto tile : discard_candidates) {
    if (!Is(tile.Type(), TileSetType::kTanyao)) continue;
    if (is_head(tile) || is_pon(tile) || is_chi(tile)) continue;
    return tile;
  }
  // 上記以外のときは、ランダムに切る
  return *SelectRandomly(discard_candidates.begin(), discard_candidates.end(),
                         g);
}
}  // namespace mjx::internal
