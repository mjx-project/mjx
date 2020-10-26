#ifndef MAHJONG_AGENT_H
#define MAHJONG_AGENT_H

#include "action.h"
#include "observation.h"

namespace mj
{
    class Agent
    {
    public:
        Agent() = default;  // generate invalid object
        explicit Agent(PlayerId player_id);
        virtual ~Agent() = default;
        [[nodiscard]] virtual Action TakeAction(Observation &&observation) const = 0;
        [[nodiscard]] PlayerId player_id() const;
    private:
        PlayerId player_id_;
    };
}  // namespace mj

#endif //MAHJONG_AGENT_H
