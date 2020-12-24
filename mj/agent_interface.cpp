#include "agent_interface.h"
#include "utils.h"

#include <utility>

namespace mj
{
    AgentInterface::AgentInterface(PlayerId player_id): player_id_(std::move(player_id)) {}

    PlayerId AgentInterface::player_id() const {
        Assert(!player_id_.empty());
        return player_id_;
    }
}  // namespace mj

