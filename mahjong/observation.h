#ifndef MAHJONG_OBSERVATION_H
#define MAHJONG_OBSERVATION_H

#include <mahjong.pb.h>

namespace mj
{
    class Observation
    {
    public:
        Observation() = default;
        ActionRequest GetActionRequest();
    private:
        ActionRequest action_request_;
    };
}

#endif //MAHJONG_OBSERVATION_H
