#ifndef MAHJONG_OBSERVATION_H
#define MAHJONG_OBSERVATION_H

#include <mahjong.pb.h>

#include <utility>

namespace mj
{
    class Observation
    {
    public:
        // As Observation should return the ownership of common observation. The life span of Observation should be short.
        Observation() = default;
        ~Observation() {
            std::cout << "Observation destructor is called" << std::endl;
            // Observation borrowed common observation when constructed. So it should return its ownership.
            common_observation_ = action_request_.release_common_observation();
        }
        Observation(ActionRequest &action_request, ActionRequest_CommonObservation* common_observation)
        : action_request_(action_request), common_observation_(common_observation) {
            action_request_.set_allocated_common_observation(common_observation_);
        }
        [[nodiscard]] const ActionRequest& GetActionRequest() const { return action_request_; }
        std::string ToString() const;
    private:
        ActionRequest &action_request_;
        ActionRequest_CommonObservation* common_observation_;
    };
}

#endif //MAHJONG_OBSERVATION_H
