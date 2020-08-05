#include "observation.h"
#include "utils.h"

namespace mj
{
    Score::Score()
    {
        score_.set_round(0);
        score_.set_honba(0);
        score_.set_riichi(0);
        for (int i = 0; i < 4; ++i) score_.add_ten(25000);
    }

    std::uint8_t Score::round() const {
        return score_.round();
    }

    std::uint8_t Score::honba() const {
        return score_.honba();
    }

    std::uint8_t Score::riichi() const {
        return score_.riichi();
    }

    std::array<std::int32_t, 4> Score::ten() const {
        assert(score_.ten_size() == 4);
        auto ret = std::array<std::int32_t, 4>();
        for (int i = 0; i < 4; ++i) ret[i] = score_.ten(i);
        return ret;
    }

    Score::Score(mjproto::Score score) {
        score_ = score;
    }

    void Score::Riichi(AbsolutePos who) {
        score_.set_riichi(riichi() + 1);
        score_.set_ten(ToUType(who), score_.ten(ToUType(who)) - 1000);
    }

    PossibleAction::PossibleAction(mjproto::PossibleAction possible_action)
    : possible_action_(std::move(possible_action)) {}

    ActionType PossibleAction::type() const {
        return ActionType(possible_action_.type());
    }

    Open PossibleAction::open() const {
        return Open(possible_action_.open());
    }

    std::vector<Tile> PossibleAction::discard_candidates() const {
        std::vector<Tile> ret;
        for (const auto& id: possible_action_.discard_candidates()) ret.emplace_back(Tile(id));
        return ret;
    }

    PossibleAction PossibleAction::CreateDiscard(const std::vector<Tile>& possible_discards) {
        auto possible_action = PossibleAction();
        possible_action.possible_action_.set_type(ToUType(ActionType::kDiscard));
        auto discard_candidates = possible_action.possible_action_.mutable_discard_candidates();
        for (auto tile: possible_discards) discard_candidates->Add(tile.Id());
        assert(discard_candidates->size() <= 14);
        return possible_action;
    }

    std::vector<PossibleAction> Observation::possible_actions() const {
        std::vector<PossibleAction> ret;
        for (const auto& possible_action: proto_.possible_actions()) {
            ret.emplace_back(PossibleAction{possible_action});
        }
        return ret;
    }

    AbsolutePos Observation::who() const {
        return AbsolutePos(proto_.who());
    }

    void Observation::ClearPossibleActions() {
        proto_.clear_possible_actions();
    }

    Observation::~Observation() {
        // Calling release_xxx prevent gRPC from deleting objects after gRPC communication
        proto_.release_init_score();
        proto_.release_event_history();
    }

    void Observation::add_possible_action(PossibleAction possible_action) {
        // TDOO (sotetsuk): add assertion. もしtypeがdiscardならすでにあるpossible_actionはdiscardではない
        auto mutable_possible_actions = proto_.mutable_possible_actions();
        mutable_possible_actions->Add(std::move(possible_action.possible_action_));
    }

    Observation::Observation(AbsolutePos who, Score &score, Player& player) {
        proto_.set_who(mjproto::AbsolutePos(ToUType(who)));
        proto_.set_allocated_init_score(&score.score_);
    }
}
