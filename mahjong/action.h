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
        Action() = delete;
        explicit Action(mjproto::ActionResponse action_response) : action_response_(std::move(action_response)) {}
        AbsolutePos who() const { return AbsolutePos(action_response_.who()); }
        ActionType type() const { return ActionType(action_response_.type()); }
        bool yes() const { return action_response_.yes(); }
        Tile discard() const {return Tile(action_response_.discard()); }
        std::unique_ptr<Open> open() const { return Open::NewOpen(action_response_.open()); }
        [[nodiscard]] const mjproto::ActionResponse& action_response() const { return action_response_; }
    private:
        mjproto::ActionResponse action_response_;
    };
}  // namespace mj

#endif //MAHJONG_ACTION_H
