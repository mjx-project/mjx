#include "agent.h"

#include <boost/assert.hpp>
#include <utility>

namespace mj
{
    Agent::Agent(PlayerId player_id): player_id_(std::move(player_id)) {}

    PlayerId Agent::player_id() const {
        BOOST_ASSERT(!player_id_.empty());
        return player_id_;
    }
}  // namespace mj

