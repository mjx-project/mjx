#include "agent_example_rule_based.h"
#include "strategy_rule_based.h"
#include "utils.h"

namespace mj
{
    AgentExampleRuleBased::AgentExampleRuleBased(PlayerId player_id) : Agent(std::move(player_id)) {}

    mjproto::Action AgentExampleRuleBased::TakeAction(Observation &&observation) const {
        return StrategyRuleBased::TakeAction(std::forward<Observation>(observation));
    }
}
