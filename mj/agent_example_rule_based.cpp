#include "agent_example_rule_based.h"
#include "utils.h"

namespace mj
{
    AgentExampleRuleBased::AgentExampleRuleBased(PlayerId player_id) : Agent(std::move(player_id)) {}

    mjproto::Action AgentExampleRuleBased::TakeAction(Observation &&observation) const {
        // Prepare some seed and MT engine for reproducibility
        const std::uint64_t seed = 12345
                + 4096 * observation.proto().event_history().events_size()
                + 16 * observation.possible_actions().size()
                + 1 * observation.proto().who();
        auto mt = std::mt19937_64(seed);

        auto possible_actions = observation.possible_actions();
        // この順番でソート
        std::unordered_map<mjproto::ActionType, int> action_priority = {
                {mjproto::ACTION_TYPE_TSUMO, 0},
                {mjproto::ACTION_TYPE_RIICHI, 1},
                {mjproto::ACTION_TYPE_KYUSYU, 2},
                {mjproto::ACTION_TYPE_KAN_CLOSED, 3},
                {mjproto::ACTION_TYPE_KAN_ADDED, 4},
                {mjproto::ACTION_TYPE_DISCARD, 5},
                {mjproto::ACTION_TYPE_RON, 6},
                {mjproto::ACTION_TYPE_PON, 7},
                {mjproto::ACTION_TYPE_KAN_OPENED, 8},
                {mjproto::ACTION_TYPE_CHI, 9},
                {mjproto::ACTION_TYPE_NO, 10},
        };
        std::sort(possible_actions.begin(), possible_actions.end(),
                [&action_priority](const mjproto::Action &x, const mjproto::Action &y){
            if (x.type() != y.type()) return action_priority.at(x.type()) < action_priority.at(y.type());
            else return x.open() < y.open();
        });

        const Hand curr_hand = observation.current_hand();

        auto& selected = possible_actions.front();

        // 和了れるときは全て和了る。リーチできるときは全てリーチする。九種九牌も全て流す。
        if (Any(selected.type(), {mjproto::ACTION_TYPE_TSUMO, mjproto::ACTION_TYPE_RIICHI,
                                  mjproto::ACTION_TYPE_RON, mjproto::ACTION_TYPE_KYUSYU})) {
            return selected;
        }

        // テンパっているときには他家から鳴かない
        if (Any(selected.type(), {mjproto::ACTION_TYPE_KAN_OPENED, mjproto::ACTION_TYPE_PON,
                                         mjproto::ACTION_TYPE_CHI})) {
            if (curr_hand.IsTenpai()) {
                selected = *possible_actions.rbegin();
                Assert(selected.type() == mjproto::ActionType::ACTION_TYPE_NO);
                return selected;
            }
        }

        // 鳴ける場合にはランダムに行動選択
        if (Any(selected.type(), {mjproto::ACTION_TYPE_KAN_OPENED, mjproto::ACTION_TYPE_PON, mjproto::ACTION_TYPE_CHI})) {
            selected = *SelectRandomly(possible_actions.begin(), possible_actions.end(), mt);
            Assert(Any(selected.type(), {
                    mjproto::ACTION_TYPE_KAN_OPENED, mjproto::ACTION_TYPE_PON,
                    mjproto::ACTION_TYPE_CHI, mjproto::ACTION_TYPE_NO}));
            return selected;
        }

        // 暗槓、加槓ができるときはランダムに実行
        if (Any(selected.type(), {mjproto::ACTION_TYPE_KAN_CLOSED, mjproto::ACTION_TYPE_KAN_ADDED})) {
            auto prob = RandomProb(mt);
            if (prob > 0.5) return selected;
        }

        // Discardが選択されたとき（あるいはdiscardしかできないとき）、切る牌を選ぶ
        std::vector<Tile> discard_candidates = observation.possible_discards();
        Tile selected_discard = SelectDiscard(discard_candidates, curr_hand, mt);
        return *std::find_if(possible_actions.begin(), possible_actions.end(),
                [&selected_discard](const mjproto::Action& x){ return x.discard() == selected_discard.Id(); });
    }

    template<typename RandomGenerator>
    Tile AgentExampleRuleBased::SelectDiscard(std::vector<Tile> &discard_candidates, const Hand &curr_hand, RandomGenerator& g) const {
        std::sort(discard_candidates.begin(), discard_candidates.end());
        const TileTypeCount closed_tile_type_cnt = curr_hand.ClosedTileTypes();
        // 聴牌が取れるなら取れるように切る
        if (curr_hand.CanTakeTenpai()) {
            auto tenpai_discards = curr_hand.PossibleDiscardsToTakeTenpai();
            for (const auto tile: discard_candidates) {
                if (Any(tile, tenpai_discards)) {
                    return tile;
                }
            }
            Assert(false);
        }
        // 判定ロジック
        auto is_head = [&closed_tile_type_cnt](Tile tile){
            return closed_tile_type_cnt.count(tile.Type()) && closed_tile_type_cnt.at(tile.Type()) >= 2;
        };
        auto is_pon = [&closed_tile_type_cnt](Tile tile){
            return closed_tile_type_cnt.count(tile.Type()) && closed_tile_type_cnt.at(tile.Type()) >= 3;
        };
        auto has_next = [&closed_tile_type_cnt](Tile tile, int n){
            return (bool) closed_tile_type_cnt.count(TileType(ToUType(tile.Type()) + n));
        };
        auto has_prev = [&closed_tile_type_cnt](Tile tile, int n){
            return (bool) closed_tile_type_cnt.count(TileType(ToUType(tile.Type()) - n));
        };
        auto is_chi = [&](Tile tile){
            if (Any(tile.Type(), {TileType::kM1, TileType::kP1, TileType::kS1}))
                return has_next(tile, 1) && has_next(tile, 2);
            if (Any(tile.Type(), {TileType::kM2, TileType::kP2, TileType::kS2}))
                return (has_next(tile, 1) && has_next(tile, 2)) || (has_prev(tile, 1) && has_next(tile, 1));
            if (Any(tile.Type(), {TileType::kM9, TileType::kP9, TileType::kS9}))
                return has_prev(tile, 1) && has_prev(tile, 2);
            if (Any(tile.Type(), {TileType::kM8, TileType::kP8, TileType::kS8}))
                return (has_prev(tile, 1) && has_prev(tile, 2)) || (has_prev(tile, 1) && has_next(tile, 1));
            return (has_next(tile, 1) && has_next(tile, 2)) || (has_prev(tile, 1) && has_prev(tile, 2)) || (has_prev(tile, 1) && has_next(tile, 1));
        };
        auto has_neighbors = [&](Tile tile){
            if (Any(tile.Type(), {TileType::kM1, TileType::kP1, TileType::kS1})) return has_next(tile, 1);
            if (Any(tile.Type(), {TileType::kM9, TileType::kP9, TileType::kS9})) return has_prev(tile, 1);
            return has_next(tile, 1) || has_prev(tile, 1);
        };
        auto has_skip_neighbors = [&](Tile tile){
            if (Any(tile.Type(), {TileType::kM1, TileType::kP1, TileType::kS1, TileType::kM2, TileType::kP2, TileType::kS2})) return has_next(tile, 2);
            if (Any(tile.Type(), {TileType::kM9, TileType::kP9, TileType::kS9, TileType::kM8, TileType::kP8, TileType::kS8})) return has_prev(tile, 2);
            return has_next(tile, 2) || has_prev(tile, 2);
        };
        // 字牌孤立牌があればまずそれを切り飛ばす
        for (const auto tile: discard_candidates) {
            if (!Is(tile.Type(), TileSetType::kHonours)) continue;
            if (is_head(tile) || is_pon(tile)) continue;
            return tile;
        }
        // 19の孤立牌を切り飛ばす
        for (const auto tile: discard_candidates) {
            if (!Is(tile.Type(), TileSetType::kTerminals)) continue;
            if (is_head(tile) || is_pon(tile) || is_chi(tile) || has_neighbors(tile) || has_skip_neighbors(tile)) continue;
            return tile;
        }
        // 断么九の孤立牌を切り飛ばす
        for (const auto tile: discard_candidates) {
            if (!Is(tile.Type(), TileSetType::kTanyao)) continue;
            if (is_head(tile) || is_pon(tile) || is_chi(tile) || has_neighbors(tile) || has_skip_neighbors(tile)) continue;
            return tile;
        }
        // 19ペンチャンを外す
        for (const auto tile: discard_candidates) {
            if (!Is(tile.Type(), TileSetType::kTerminals)) continue;
            if (is_head(tile) || is_pon(tile) || is_chi(tile) || has_skip_neighbors(tile)) continue;
            return tile;
        }
        // 19カンチャンを外す
        for (const auto tile: discard_candidates) {
            if (!Is(tile.Type(), TileSetType::kTerminals)) continue;
            if (is_head(tile) || is_pon(tile) || is_chi(tile) || has_neighbors(tile)) continue;
            return tile;
        }
        // 断么九のカンチャンを外す
        for (const auto tile: discard_candidates) {
            if (!Is(tile.Type(), TileSetType::kTanyao)) continue;
            if (is_head(tile) || is_pon(tile) || is_chi(tile) || has_neighbors(tile)) continue;
            return tile;
        }
        // 断么九の両面を外す
        for (const auto tile: discard_candidates) {
            if (!Is(tile.Type(), TileSetType::kTanyao)) continue;
            if (is_head(tile) || is_pon(tile) || is_chi(tile)) continue;
            return tile;
        }
        // 上記以外のときは、ランダムに切る
        return *SelectRandomly(discard_candidates.begin(), discard_candidates.end(), g);
    }
}
