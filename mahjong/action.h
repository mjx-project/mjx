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
        AbsolutePos who() const;
        ActionType type() const;
        bool yes() const;
        Tile discard() const;
        std::unique_ptr<Open> open() const;
        [[nodiscard]] const ActionResponse& action_response() const;
    private:
        ActionResponse action_response_;
    };
}  // namespace mj

#endif //MAHJONG_ACTION_H
