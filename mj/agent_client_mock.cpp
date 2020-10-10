#include "agent_client_mock.h"
#include "utils.h"

namespace mj
{
    Action AgentClientMock::TakeAction(Observation &&observation) const {
        // Currently this method only implements discard
        mjproto::Action response;
        response.set_who(mjproto::AbsolutePos(observation.who()));
        auto possible_actions = observation.possible_actions();
        // 和了れるときは全て和了り、鳴けるときは全て鳴く方針
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
        const auto& possible_action = possible_actions.front();

        // 和了れるときは全て和了り、鳴けるときは全て鳴く
        if (possible_action.type() != ActionType::kDiscard) {
            response.set_type(mjproto::ActionType(possible_action.type()));
            if (Any(possible_action.type(),
                    {ActionType::kKanClosed, ActionType::kKanAdded, ActionType::kKanOpened, ActionType::kPon,
                     ActionType::kChi})) {
                response.set_open(possible_action.open().GetBits());
            }
            return Action(std::move(response));
        }

        // discardしか取れるアクションがないとき
        const Hand curr_hand = observation.current_hand();
        const TileTypeCount closed_tile_type_cnt = curr_hand.ClosedTileTypes();
        // 聴牌が取れるなら取れるように切る
        if (curr_hand.CanTakeTenpai()) {
            auto tile = curr_hand.PossibleDiscardsToTakeTenpai().front();
            response.set_discard(tile.Id());
            return Action(std::move(response));
        }
        // 字牌孤立牌があればまずそれを切り飛ばす
        for (const auto tile: possible_action.discard_candidates()) {
            if (!Is(tile.Type(), TileSetType::kHonours)) continue;
            if (closed_tile_type_cnt.count(tile.Type()) && closed_tile_type_cnt.at(tile.Type()) >= 2) continue;
            response.set_discard(tile.Id());
            return Action(std::move(response));
        }
        // 19の孤立牌を切り飛ばす
        auto is_independent = [&closed_tile_type_cnt](Tile tile){
            if (closed_tile_type_cnt.count(tile.Type()) && closed_tile_type_cnt.at(tile.Type()) >= 2) return false;
            if (!Any(tile.Type(), {TileType::kM1, TileType::kP1, TileType::kS1}) && closed_tile_type_cnt.count(TileType(ToUType(tile.Type()) - 1))) return false;
            if (!Any(tile.Type(), {TileType::kM9, TileType::kP9, TileType::kS9}) && closed_tile_type_cnt.count(TileType(ToUType(tile.Type()) + 1))) return false;
            return true;
        };
        for (const auto tile: possible_action.discard_candidates()) {
            if (!Is(tile.Type(), TileSetType::kTerminals)) continue;
            if (is_independent(tile)) continue;
            response.set_discard(tile.Id());
            return Action(std::move(response));
        }
        // 断么九の孤立牌を切り飛ばす
        for (const auto tile: possible_action.discard_candidates()) {
            if (!Is(tile.Type(), TileSetType::kTanyao)) continue;
            if (is_independent(tile)) continue;
            response.set_discard(tile.Id());
            return Action(std::move(response));
        }
        // 上記以外のときは、ランダムに切る
        response.set_discard(SelectRandomly(possible_action.discard_candidates().begin(), possible_action.discard_candidates().end())->Id());
        return Action(std::move(response));
    }
}
