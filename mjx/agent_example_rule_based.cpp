#include "agent_example_rule_based.h"
#include "strategy_rule_based.h"
#include "utils.h"

namespace mjx
{
    AgentExampleRuleBased::AgentExampleRuleBased(PlayerId player_id) : Agent(std::move(player_id)) {}

    mjxproto::Action AgentExampleRuleBased::TakeAction(Observation &&observation) const {
        return strategy_.TakeAction(std::forward<Observation>(observation));
    }
}
