#include "action.h"
#include "utils.h"
#include "mjx.grpc.pb.h"

namespace mjx
{
    mjproto::Action Action::CreateDiscard(AbsolutePos who, Tile discard) {
        mjproto::Action proto;
        proto.set_type(mjproto::ACTION_TYPE_DISCARD);
        proto.set_who(mjproto::AbsolutePos(who));
        proto.set_discard(discard.Id());
        Assert(IsValid(proto));
        return proto;
    }

    std::vector<mjproto::Action> Action::CreateDiscards(AbsolutePos who, const std::vector<Tile>& discards) {
        std::vector<mjproto::Action> ret;
        for (auto tile : discards) {
            ret.push_back(CreateDiscard(who, tile));
        }
        return ret;
    }

    mjproto::Action Action::CreateRiichi(AbsolutePos who) {
        mjproto::Action proto;
        proto.set_type(mjproto::ACTION_TYPE_RIICHI);
        proto.set_who(mjproto::AbsolutePos(who));
        Assert(IsValid(proto));
        return proto;
    }

    mjproto::Action Action::CreateTsumo(AbsolutePos who) {
        mjproto::Action proto;
        proto.set_type(mjproto::ACTION_TYPE_TSUMO);
        proto.set_who(mjproto::AbsolutePos(who));
        Assert(IsValid(proto));
        return proto;
    }

    mjproto::Action Action::CreateRon(AbsolutePos who) {
        mjproto::Action proto;
        proto.set_type(mjproto::ACTION_TYPE_RON);
        proto.set_who(mjproto::AbsolutePos(who));
        Assert(IsValid(proto));
        return proto;
    }

    mjproto::Action Action::CreateOpen(AbsolutePos who, Open open) {
        mjproto::Action proto;
        proto.set_who(mjproto::AbsolutePos(who));
        proto.set_type(OpenTypeToActionType(open.Type()));
        proto.set_open(open.GetBits());
        Assert(IsValid(proto));
        return proto;
    }

    mjproto::Action Action::CreateNo(AbsolutePos who) {
        mjproto::Action proto;
        proto.set_type(mjproto::ACTION_TYPE_NO);
        proto.set_who(mjproto::AbsolutePos(who));
        Assert(IsValid(proto));
        return proto;
    }

    mjproto::Action Action::CreateNineTiles(AbsolutePos who) {
        mjproto::Action proto;
        proto.set_type(mjproto::ACTION_TYPE_KYUSYU);
        proto.set_who(mjproto::AbsolutePos(who));
        Assert(IsValid(proto));
        return proto;
    }

    bool Action::IsValid(const mjproto::Action &action) {
        auto type = action.type();
        auto who = action.who();
        if (!mjproto::AbsolutePos_IsValid(who)) return false;
        switch (type) {
            case mjproto::ACTION_TYPE_DISCARD:
                if (!(0 <= action.discard() && action.discard() < 136)) return false;
                if (action.open() != 0) return false;
                break;
            case mjproto::ACTION_TYPE_CHI:
            case mjproto::ACTION_TYPE_PON:
            case mjproto::ACTION_TYPE_KAN_CLOSED:
            case mjproto::ACTION_TYPE_KAN_ADDED:
            case mjproto::ACTION_TYPE_KAN_OPENED:
                if (action.discard() != 0) return false;
                break;
            case mjproto::ACTION_TYPE_RIICHI:
            case mjproto::ACTION_TYPE_TSUMO:
            case mjproto::ACTION_TYPE_KYUSYU:
            case mjproto::ACTION_TYPE_NO:
            case mjproto::ACTION_TYPE_RON:
                if (action.discard() != 0) return false;
                if (action.open() != 0) return false;
                break;
        }
        return true;
    }
    bool Action::Equal(const mjproto::Action& lhs, const mjproto::Action& rhs) {
        return lhs.game_id() == rhs.game_id() and
               lhs.who() == rhs.who() and
               lhs.type() == rhs.type() and
               lhs.discard() == rhs.discard() and
               lhs.open() == rhs.open();
    }
}  // namespace mjx
