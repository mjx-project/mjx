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
        assert(type() == mjproto::ActionType::ACTION_TYPE_DISCARD);
        return Tile(proto_.discard());
    }

    Open Action::open() const {
        assert(Any(type(), {mjproto::ActionType::ACTION_TYPE_CHI, mjproto::ActionType::ACTION_TYPE_PON,
                            mjproto::ActionType::ACTION_TYPE_KAN_CLOSED, mjproto::ActionType::ACTION_TYPE_KAN_OPENED,
                            mjproto::ActionType::ACTION_TYPE_KAN_ADDED}));
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

    Open PossibleAction::open() const {
        assert(Any(type(), {mjproto::ActionType::ACTION_TYPE_CHI, mjproto::ActionType::ACTION_TYPE_PON,
                            mjproto::ActionType::ACTION_TYPE_KAN_CLOSED, mjproto::ActionType::ACTION_TYPE_KAN_OPENED,
                            mjproto::ActionType::ACTION_TYPE_KAN_ADDED}));
        return Open(possible_action_.open());
    }

    std::vector<Tile> PossibleAction::discard_candidates() const {
        assert(type() == mjproto::ActionType::ACTION_TYPE_DISCARD);
        std::vector<Tile> ret;
        for (const auto& id: possible_action_.discard_candidates()) ret.emplace_back(Tile(id));
        return ret;
    }

    PossibleAction PossibleAction::CreateDiscard(std::vector<Tile> &&possible_discards) {
        auto possible_action = PossibleAction();
        possible_action.possible_action_.set_type(mjproto::ActionType::ACTION_TYPE_DISCARD);
        auto discard_candidates = possible_action.possible_action_.mutable_discard_candidates();
        for (auto tile: possible_discards) discard_candidates->Add(tile.Id());
        assert(discard_candidates->size() <= 14);
        return possible_action;
    }

    PossibleAction PossibleAction::CreateRiichi() {
        auto possible_action = PossibleAction();
        possible_action.possible_action_.set_type(mjproto::ActionType::ACTION_TYPE_RIICHI);
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
        possible_action.possible_action_.set_type(mjproto::ActionType::ACTION_TYPE_RON);
        return possible_action;
    }

    PossibleAction PossibleAction::CreateTsumo() {
        auto possible_action = PossibleAction();
        possible_action.possible_action_.set_type(mjproto::ActionType::ACTION_TYPE_TSUMO);
        return possible_action;
    }

    PossibleAction PossibleAction::CreateNo() {
        auto possible_action = PossibleAction();
        possible_action.possible_action_.set_type(mjproto::ActionType::ACTION_TYPE_NO);
        return possible_action;
    }

    PossibleAction PossibleAction::CreateNineTiles() {
        auto possible_action = PossibleAction();
        possible_action.possible_action_.set_type(mjproto::ActionType::ACTION_TYPE_KYUSYU);
        return possible_action;
    }

    std::string PossibleAction::ToJson() const {
        std::string serialized;
        auto status = google::protobuf::util::MessageToJsonString(possible_action_, &serialized);
        assert(status.ok());
        return serialized;
    }
}  // namespace mj
