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

    PossibleAction::PossibleAction(mjproto::PossibleAction possible_action)
            : possible_action_(std::move(possible_action)) {}

    mjproto::ActionType PossibleAction::type() const {
        return possible_action_.type();
    }

    Tile PossibleAction::discard() const {
        return static_cast<Tile>(possible_action_.discard());
    }

    Open PossibleAction::open() const {
        assert(Any(type(), {mjproto::ACTION_TYPE_CHI, mjproto::ACTION_TYPE_PON,
                            mjproto::ACTION_TYPE_KAN_CLOSED, mjproto::ACTION_TYPE_KAN_OPENED,
                            mjproto::ACTION_TYPE_KAN_ADDED}));
        return Open(possible_action_.open());
    }

    std::vector<PossibleAction> PossibleAction::CreateDiscard(const std::vector<Tile> &possible_discards) {
        std::vector<PossibleAction> ret;
        for (const auto& tile : possible_discards) {
            ret.push_back(CreateDiscard(tile));
        }
        return ret;
    }

    PossibleAction PossibleAction::CreateDiscard(Tile possible_discard) {
        PossibleAction action;
        action.possible_action_.set_type(mjproto::ACTION_TYPE_DISCARD);
        action.possible_action_.set_discard(possible_discard.Id());
        return action;
    }

    PossibleAction PossibleAction::CreateRiichi() {
        auto possible_action = PossibleAction();
        possible_action.possible_action_.set_type(mjproto::ACTION_TYPE_RIICHI);
        return possible_action;
    }

    PossibleAction PossibleAction::CreateOpen(Open open) {
        auto possible_action = PossibleAction();
        possible_action.possible_action_.set_type(OpenTypeToActionType(open.Type()));
        possible_action.possible_action_.set_open(open.GetBits());
        return possible_action;
    }

    PossibleAction PossibleAction::CreateRon() {
        auto possible_action = PossibleAction();
        possible_action.possible_action_.set_type(mjproto::ACTION_TYPE_RON);
        return possible_action;
    }

    PossibleAction PossibleAction::CreateTsumo() {
        auto possible_action = PossibleAction();
        possible_action.possible_action_.set_type(mjproto::ACTION_TYPE_TSUMO);
        return possible_action;
    }

    PossibleAction PossibleAction::CreateNo() {
        auto possible_action = PossibleAction();
        possible_action.possible_action_.set_type(mjproto::ACTION_TYPE_NO);
        return possible_action;
    }

    PossibleAction PossibleAction::CreateNineTiles() {
        auto possible_action = PossibleAction();
        possible_action.possible_action_.set_type(mjproto::ACTION_TYPE_KYUSYU);
        return possible_action;
    }

    std::string PossibleAction::ToJson() const {
        std::string serialized;
        auto status = google::protobuf::util::MessageToJsonString(possible_action_, &serialized);
        assert(status.ok());
        return serialized;
    }
}  // namespace mj
