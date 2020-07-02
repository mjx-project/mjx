#include "action.h"

namespace mj
{
    AbsolutePos Action::who() const {
        return AbsolutePos(action_response_.who());
    }

    ActionType Action::type() const {
        return ActionType(action_response_.type());
    }

    bool Action::yes() const {
        return action_response_.yes();
    }

    Tile Action::discard() const {
        return Tile(action_response_.discard());
    }

    std::unique_ptr<Open> Action::open() const {
        return Open::NewOpen(action_response_.open());
    }

    const ActionResponse& Action::action_response() const {
        return action_response_;
    }
}
