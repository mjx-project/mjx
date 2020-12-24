#ifndef MAHJONG_AGENT_INTERFACE_H
#define MAHJONG_AGENT_INTERFACE_H

#include "action.h"
#include "observation.h"

namespace mj
{
    class AgentInterface
    {
    public:
        AgentInterface() = default;  // generate invalid object
        explicit AgentInterface(PlayerId player_id);
        virtual ~AgentInterface() = default;
        [[nodiscard]] virtual mjproto::Action TakeAction(Observation &&observation) const = 0;
        [[nodiscard]] PlayerId player_id() const;
    private:
        PlayerId player_id_;
    };
}  // namespace mj

#endif //MAHJONG_AGENT_INTERFACE_H
