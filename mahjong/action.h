#ifndef MAHJONG_ACTION_H
#define MAHJONG_ACTION_H

#include <memory>
#include <utility>
#include <mahjong.pb.h>
#include "types.h"
#include "tile.h"
#include "open.h"

namespace mj
{
    class Action
    {
    public:
        Action() = default;
        Action(ActionResponse action_response)
        : action_response_(std::move(action_response)) {}
        AbsolutePos GetWho() const;
        ActionType GetType() const;
        bool GetYes() const;
        Tile GetDiscard() const;
        std::unique_ptr<Open> GetOpen() const;
        [[nodiscard]] const ActionResponse& GetActionResponse() const { return action_response_; }
    private:
        ActionResponse action_response_;
    };
}  // namespace mj

#endif //MAHJONG_ACTION_H
