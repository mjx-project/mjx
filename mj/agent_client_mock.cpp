#include "agent_client_mock.h"
#include "utils.h"

namespace mj
{
    Action AgentClientMock::TakeAction(Observation &&observation) const {
        // Currently this method only implements discard
        mjproto::Action response;
        response.set_who(mjproto::AbsolutePos(observation.who()));
        auto possible_actions = observation.possible_actions();
        // 和了れるときは全て和了り、鳴けるときは全て鳴く
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
        if (possible_action.type() == ActionType::kDiscard) {
            // TODO: implement here
        } else {
            response.set_type(mjproto::ActionType(possible_action.type()));
            if (Any(possible_action.type(),{ActionType::kKanClosed, ActionType::kKanAdded, ActionType::kKanOpened, ActionType::kPon, ActionType::kChi})) {
                response.set_open(possible_action.open().GetBits());
            }
        }
        return Action(std::move(response));
    }
}
