#include "action.h"
#include "utils.h"
#include "mjx.grpc.pb.h"

namespace mjx
{
    mjxproto::Action Action::CreateDiscard(AbsolutePos who, Tile discard) {
        mjxproto::Action proto;
        proto.set_type(mjxproto::ACTION_TYPE_DISCARD);
        proto.set_who(mjxproto::AbsolutePos(who));
        proto.set_discard(discard.Id());
        Assert(IsValid(proto));
        return proto;
    }

    std::vector<mjxproto::Action> Action::CreateDiscards(AbsolutePos who, const std::vector<Tile>& discards) {
        std::vector<mjxproto::Action> ret;
        for (auto tile : discards) {
            ret.push_back(CreateDiscard(who, tile));
        }
        return ret;
    }

    mjxproto::Action Action::CreateRiichi(AbsolutePos who) {
        mjxproto::Action proto;
        proto.set_type(mjxproto::ACTION_TYPE_RIICHI);
        proto.set_who(mjxproto::AbsolutePos(who));
        Assert(IsValid(proto));
        return proto;
    }

    mjxproto::Action Action::CreateTsumo(AbsolutePos who) {
        mjxproto::Action proto;
        proto.set_type(mjxproto::ACTION_TYPE_TSUMO);
        proto.set_who(mjxproto::AbsolutePos(who));
        Assert(IsValid(proto));
        return proto;
    }

    mjxproto::Action Action::CreateRon(AbsolutePos who) {
        mjxproto::Action proto;
        proto.set_type(mjxproto::ACTION_TYPE_RON);
        proto.set_who(mjxproto::AbsolutePos(who));
        Assert(IsValid(proto));
        return proto;
    }

    mjxproto::Action Action::CreateOpen(AbsolutePos who, Open open) {
        mjxproto::Action proto;
        proto.set_who(mjxproto::AbsolutePos(who));
        proto.set_type(OpenTypeToActionType(open.Type()));
        proto.set_open(open.GetBits());
        Assert(IsValid(proto));
        return proto;
    }

    mjxproto::Action Action::CreateNo(AbsolutePos who) {
        mjxproto::Action proto;
        proto.set_type(mjxproto::ACTION_TYPE_NO);
        proto.set_who(mjxproto::AbsolutePos(who));
        Assert(IsValid(proto));
        return proto;
    }

    mjxproto::Action Action::CreateNineTiles(AbsolutePos who) {
        mjxproto::Action proto;
        proto.set_type(mjxproto::ACTION_TYPE_KYUSYU);
        proto.set_who(mjxproto::AbsolutePos(who));
        Assert(IsValid(proto));
        return proto;
    }

    bool Action::IsValid(const mjxproto::Action &action) {
        auto type = action.type();
        auto who = action.who();
        if (!mjxproto::AbsolutePos_IsValid(who)) return false;
        switch (type) {
            case mjxproto::ACTION_TYPE_DISCARD:
                if (!(0 <= action.discard() && action.discard() < 136)) return false;
                if (action.open() != 0) return false;
                break;
            case mjxproto::ACTION_TYPE_CHI:
            case mjxproto::ACTION_TYPE_PON:
            case mjxproto::ACTION_TYPE_KAN_CLOSED:
            case mjxproto::ACTION_TYPE_KAN_ADDED:
            case mjxproto::ACTION_TYPE_KAN_OPENED:
                if (action.discard() != 0) return false;
                break;
            case mjxproto::ACTION_TYPE_RIICHI:
            case mjxproto::ACTION_TYPE_TSUMO:
            case mjxproto::ACTION_TYPE_KYUSYU:
            case mjxproto::ACTION_TYPE_NO:
            case mjxproto::ACTION_TYPE_RON:
                if (action.discard() != 0) return false;
                if (action.open() != 0) return false;
                break;
        }
        return true;
    }
    bool Action::Equal(const mjxproto::Action& lhs, const mjxproto::Action& rhs) {
        return lhs.game_id() == rhs.game_id() and
               lhs.who() == rhs.who() and
               lhs.type() == rhs.type() and
               lhs.discard() == rhs.discard() and
               lhs.open() == rhs.open();
    }
}  // namespace mjx
