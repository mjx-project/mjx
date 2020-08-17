#include "action.h"

namespace mj
{
    Action Action::CreateDiscard(AbsolutePos who, Tile discard) {
        mjproto::Action proto;
        proto.set_type(mjproto::ACTION_TYPE_DISCARD);
        proto.set_who(mjproto::AbsolutePos(who));
        proto.set_discard(discard.Id());
        return Action(std::move(proto));
    }

    Action Action::CreateRiichi(AbsolutePos who) {
        mjproto::Action proto;
        proto.set_type(mjproto::ACTION_TYPE_RIICHI);
        proto.set_who(mjproto::AbsolutePos(who));
        return Action(std::move(proto));
    }

    Action Action::CreateTsumo(AbsolutePos who) {
        mjproto::Action proto;
        proto.set_type(mjproto::ACTION_TYPE_TSUMO);
        proto.set_who(mjproto::AbsolutePos(who));
        return Action(std::move(proto));
    }

    Action Action::CreateRon(AbsolutePos who) {
        mjproto::Action proto;
        proto.set_type(mjproto::ACTION_TYPE_RON);
        proto.set_who(mjproto::AbsolutePos(who));
        return Action(std::move(proto));
    }

    Action Action::CreateOpen(AbsolutePos who, Open open) {
        mjproto::Action proto;
        proto.set_type(mjproto::ActionType(OpenTypeToActionType(open.Type())));
        proto.set_who(mjproto::AbsolutePos(who));
        return Action(std::move(proto));
    }

    Action Action::CreateNo(AbsolutePos who) {
        mjproto::Action proto;
        proto.set_type(mjproto::ACTION_TYPE_NO);
        proto.set_who(mjproto::AbsolutePos(who));
        return Action(std::move(proto));
    }
}  // namespace mj
