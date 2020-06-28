#ifndef MAHJONG_OBSERVATION_H
#define MAHJONG_OBSERVATION_H

#include <mahjong.pb.h>

#include <utility>

namespace mj
{
    class Observation
    {
    public:
        Observation() = default;
        Observation(ActionRequest action_request): action_request_(std::move(action_request)) {}
        const ActionRequest& GetActionRequest() const { return action_request_; }
        std::string ToString() const;
    private:
        ActionRequest action_request_;
    };
}

#endif //MAHJONG_OBSERVATION_H
