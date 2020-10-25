#include "agent_client_mock.h"
#include "utils.h"

namespace mj
{
    AgentClientMock::AgentClientMock(PlayerId player_id): Agent(std::move(player_id)) {}

    Action AgentClientMock::TakeAction(Observation &&observation) const {
        // Currently this method only implements discard
        mjproto::Action response;
        response.set_who(mjproto::AbsolutePos(observation.who()));
        auto possible_actions = observation.possible_actions();
        // この順番でソート
        std::unordered_map<ActionType, int> action_priority = {
                {ActionType::kTsumo, 0},
                {ActionType::kRiichi, 1},
                {ActionType::kKyushu, 2},
                {ActionType::kKanClosed, 3},
                {ActionType::kKanAdded, 4},
                {ActionType::kDiscard, 5},
                {ActionType::kRon, 6},
                {ActionType::kPon, 7},
                {ActionType::kKanOpened, 8},
                {ActionType::kChi, 9},
                {ActionType::kNo, 10},
        };
        std::sort(possible_actions.begin(), possible_actions.end(),
                [&action_priority](const PossibleAction &x, const PossibleAction &y){ return action_priority.at(x.type()) < action_priority.at(y.type()); });

        const Hand curr_hand = observation.current_hand();
        auto& possible_action = possible_actions.front();

        // 和了れるときは全て和了る。リーチできるときは全てリーチする。九種九牌も全て流す。
        if (Any(possible_action.type(), {ActionType::kTsumo, ActionType::kRiichi, ActionType::kRon, ActionType::kKyushu})) {
            response.set_type(mjproto::ActionType(possible_action.type()));
            return Action(std::move(response));
        }

        // テンパっているときには他家から鳴かない
        if (Any(possible_action.type(), {ActionType::kKanOpened, ActionType::kPon, ActionType::kChi})) {
            if (curr_hand.IsTenpai()) {
                possible_action = *possible_actions.rbegin();
                assert(possible_action.type() == ActionType::kNo);
                response.set_type(mjproto::ActionType(possible_action.type()));
                return Action(std::move(response));
            }
        }

        // 鳴ける場合にはランダムに行動選択
        if (Any(possible_action.type(), {ActionType::kKanClosed, ActionType::kKanAdded, ActionType::kKanOpened, ActionType::kPon, ActionType::kChi})) {
            possible_action = *SelectRandomly(possible_actions.begin(), possible_actions.end());
            if (possible_action.type() != ActionType::kDiscard) {
                assert(Any(possible_action.type(), {ActionType::kKanClosed, ActionType::kKanAdded, ActionType::kKanOpened, ActionType::kPon, ActionType::kChi, ActionType::kNo}));
                response.set_type(mjproto::ActionType(possible_action.type()));
                if (possible_action.type() != ActionType::kNo) response.set_open(possible_action.open().GetBits());
                return Action(std::move(response));
            }
        }

        // Discardが選択されたとき（あるいはdiscardしかできないとき）、切る牌を選ぶ
        assert(possible_action.type() == ActionType::kDiscard);
        const TileTypeCount closed_tile_type_cnt = curr_hand.ClosedTileTypes();
        // 聴牌が取れるなら取れるように切る
        if (curr_hand.CanTakeTenpai()) {
            auto tenpai_discards = curr_hand.PossibleDiscardsToTakeTenpai();
            for (const auto tile: possible_action.discard_candidates()) {
                if (Any(tile, tenpai_discards)) {
                    response.set_discard(tile.Id());
                    return Action(std::move(response));
                }
            }
            assert(false);
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
        for (const auto tile: possible_action.discard_candidates()) {
            if (!Is(tile.Type(), TileSetType::kHonours)) continue;
            if (is_head(tile) || is_pon(tile)) continue;
            response.set_discard(tile.Id());
            return Action(std::move(response));
        }
        // 19の孤立牌を切り飛ばす
        for (const auto tile: possible_action.discard_candidates()) {
            if (!Is(tile.Type(), TileSetType::kTerminals)) continue;
            if (is_head(tile) || is_pon(tile) || is_chi(tile) || has_neighbors(tile) || has_skip_neighbors(tile)) continue;
            response.set_discard(tile.Id());
            return Action(std::move(response));
        }
        // 断么九の孤立牌を切り飛ばす
        for (const auto tile: possible_action.discard_candidates()) {
            if (!Is(tile.Type(), TileSetType::kTanyao)) continue;
            if (is_head(tile) || is_pon(tile) || is_chi(tile) || has_neighbors(tile) || has_skip_neighbors(tile)) continue;
            response.set_discard(tile.Id());
            return Action(std::move(response));
        }
        // 19ペンチャンを外す
        for (const auto tile: possible_action.discard_candidates()) {
            if (!Is(tile.Type(), TileSetType::kTerminals)) continue;
            if (is_head(tile) || is_pon(tile) || is_chi(tile) || has_skip_neighbors(tile)) continue;
            response.set_discard(tile.Id());
            return Action(std::move(response));
        }
        // 19カンチャンを外す
        for (const auto tile: possible_action.discard_candidates()) {
            if (!Is(tile.Type(), TileSetType::kTerminals)) continue;
            if (is_head(tile) || is_pon(tile) || is_chi(tile) || has_neighbors(tile)) continue;
            response.set_discard(tile.Id());
            return Action(std::move(response));
        }
        // 断么九のカンチャンを外す
        for (const auto tile: possible_action.discard_candidates()) {
            if (!Is(tile.Type(), TileSetType::kTanyao)) continue;
            if (is_head(tile) || is_pon(tile) || is_chi(tile) || has_neighbors(tile)) continue;
            response.set_discard(tile.Id());
            return Action(std::move(response));
        }
        // 断么九の両面を外す
        for (const auto tile: possible_action.discard_candidates()) {
            if (!Is(tile.Type(), TileSetType::kTanyao)) continue;
            if (is_head(tile) || is_pon(tile) || is_chi(tile)) continue;
            response.set_discard(tile.Id());
            return Action(std::move(response));
        }
        // 上記以外のときは、ランダムに切る
        auto possible_discards = possible_action.discard_candidates();
        response.set_discard(SelectRandomly(possible_discards.begin(), possible_discards.end())->Id());
        return Action(std::move(response));
    }
}
