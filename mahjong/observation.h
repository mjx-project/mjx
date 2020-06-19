#ifndef MAHJONG_OBSERVATION_H
#define MAHJONG_OBSERVATION_H

#include "state.h"

namespace mj
{
    class Observation
    {
    public:
        Observation();
        ActionRequest GetActionRequest();
    private:
        const State& state;
        ActionRequest action_request_;
    };
}

#endif //MAHJONG_OBSERVATION_H
