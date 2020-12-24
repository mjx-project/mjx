#include "agent_example_rule_based.h"
#include "agent_rule_based.h"
#include "utils.h"

namespace mj
{
    AgentExampleRuleBased::AgentExampleRuleBased(PlayerId player_id) : AgentInterface(std::move(player_id)) {}

    mjproto::Action AgentExampleRuleBased::TakeAction(Observation &&observation) const {
        return AgentRuleBased::TakeAction(std::forward<Observation>(observation));
    }
}
