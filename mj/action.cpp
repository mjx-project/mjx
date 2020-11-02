#include "action.h"
#include "utils.h"
#include "mj.grpc.pb.h"

namespace mj
{
    Action Action::CreateDiscard(AbsolutePos who, Tile discard) {
        mjproto::Action proto;
        proto.set_type(mjproto::ACTION_TYPE_DISCARD);
        proto.set_who(mjproto::AbsolutePos(who));
        proto.set_discard(discard.Id());
        return Action(std::move(proto));
    }

    std::vector<Action> Action::CreateDiscards(AbsolutePos who, const std::vector<Tile>& discards) {
        std::vector<Action> ret;
        for (auto tile : discards) {
            ret.push_back(CreateDiscard(who, tile));
        }
        return ret;
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
        proto.set_who(mjproto::AbsolutePos(who));
        proto.set_type(OpenTypeToActionType(open.Type()));
        proto.set_open(open.GetBits());
        return Action(std::move(proto));
    }

    Action Action::CreateNo(AbsolutePos who) {
        mjproto::Action proto;
        proto.set_type(mjproto::ACTION_TYPE_NO);
        proto.set_who(mjproto::AbsolutePos(who));
        return Action(std::move(proto));
    }

    mjproto::Action Action::Proto() const {
        return proto_;
    }

    Action::Action(mjproto::Action &&action_response) : proto_(std::move(action_response)) {}

    AbsolutePos Action::who() const {
        return AbsolutePos(proto_.who());
    }

    mjproto::ActionType Action::type() const {
        return proto_.type();
    }

    Tile Action::discard() const {
        assert(type() == mjproto::ACTION_TYPE_DISCARD);
        return Tile(proto_.discard());
    }

    Open Action::open() const {
        assert(Any(type(), {mjproto::ACTION_TYPE_CHI, mjproto::ACTION_TYPE_PON,
                            mjproto::ACTION_TYPE_KAN_CLOSED, mjproto::ACTION_TYPE_KAN_OPENED,
                            mjproto::ACTION_TYPE_KAN_ADDED}));
        return Open(proto_.open());
    }

    Action Action::CreateNineTiles(AbsolutePos who) {
        mjproto::Action proto;
        proto.set_type(mjproto::ACTION_TYPE_KYUSYU);
        proto.set_who(mjproto::AbsolutePos(who));
        return Action(std::move(proto));
    }
}  // namespace mj
