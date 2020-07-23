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
        explicit Action(mjproto::Action action_response) : proto_(std::move(action_response)) {}
        AbsolutePos who() const { return AbsolutePos(proto_.who()); }
        ActionType type() const { return ActionType(proto_.type()); }
        bool yes() const { return proto_.yes(); }
        Tile discard() const {return Tile(proto_.discard()); }
        std::unique_ptr<Open> open() const { return Open::NewOpen(proto_.open()); }
    private:
        mjproto::Action proto_;
    };
}  // namespace mj

#endif //MAHJONG_ACTION_H
