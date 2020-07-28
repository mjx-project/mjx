#include "state.h"
#include "utils.h"

#include <google/protobuf/util/json_util.h>

namespace mj
{
    State::State(std::uint32_t seed)
    : seed_(seed), score_(Score()), wall_(0)
    {
        // TODO (sotetsuk): shuffle seats
    }

    void State::InitRound() {
        // TODO: use seed_
        stage_ = RoundStage::kAfterDiscards;
        dealer_ = AbsolutePos(score_.round() % 4);
        drawer_ = dealer_;
        latest_discarder_ = AbsolutePos::kInitNorth;
        wall_ = Wall(score_.round());  // TODO: use seed_
        for (int i = 0; i < 4; ++i) players_[i] = Player{AbsolutePos(i), River(), Hand(wall_.initial_hand_tiles(AbsolutePos(i)))};
        action_history_ = Events();
    }

    std::uint32_t State::GenerateRoundSeed() {
        // TODO: use seed_
        std::random_device seed_gen;
        return seed_gen();
    }

    const Wall & State::wall() const {
        return wall_;
    }

    AbsolutePos State::UpdateStateByDraw() {
        assert(Any(stage_,
                   {RoundStage::kAfterDiscards,
                    RoundStage::kAfterKanClosed,
                    RoundStage::kAfterKanOpened,
                    RoundStage::kAfterKanAdded}));

        mutable_player(drawer_).Draw(wall_.Draw());
        // TODO (sotetsuk): update action history
        stage_ = RoundStage::kAfterDraw;
        return drawer_;
    }

    void State::UpdateStateByAction(const Action &action) {
        switch (action.type()) {
            case ActionType::kDiscard:
                mutable_player(action.who()).Discard(action.discard());
                stage_ = RoundStage::kAfterDiscards;
                drawer_ = AbsolutePos((static_cast<int>(action.who()) + 1) % 4);
                latest_discarder_ = action.who();
                break;
            default:
                static_assert(true, "Not implemented error.");
        }
    }

    RoundStage State::stage() const {
        return stage_;
    }

    bool State::IsRoundOver() {
        if (!wall_.HasDrawLeft()) return true;
        return false;
    }

    const Player &State::player(AbsolutePos pos) const {
        return players_.at(ToUType(pos));
    }

    Player& State::mutable_player(AbsolutePos pos) {
        return players_.at(ToUType(pos));
    }

    const Hand & State::hand(AbsolutePos pos) const {
        return player(pos).hand();
    }

    const River &State::river(AbsolutePos pos) const {
        return players_.at(ToUType(pos)).river();
    }

    Observation State::CreateObservation(AbsolutePos pos) {
        auto observation = Observation(pos, score_, action_history_, mutable_player(pos));
        switch (stage()) {
            case RoundStage::kAfterDraw:
                assert(hand(pos).Stage() == HandStage::kAfterDraw);
                observation.add_possible_action(PossibleAction::CreateDiscard(hand(pos)));
                // TODO(sotetsuk): add kan_added, kan_closed and riichi
                break;
            default:
                break;
        }
        return observation;
    }

    std::optional<std::vector<AbsolutePos>> State::RonCheck() {
        auto possible_winners = std::make_optional<std::vector<AbsolutePos>>();
        auto position = AbsolutePos((ToUType(latest_discarder_) + 1) % 4);
        auto discarded_tile = player(latest_discarder_).latest_discard();
        while (position != latest_discarder_) {
            if (player(position).CanRon(discarded_tile)) possible_winners->emplace_back(position);
            position = AbsolutePos((ToUType(position) + 1) % 4);
        }
        if (possible_winners.value().empty()) possible_winners = std::nullopt;
        return possible_winners;
    }

    std::optional<std::vector<std::pair<AbsolutePos, std::vector<Open>>>> State::StealCheck() {
        assert(stage() == RoundStage::kAfterDiscards);
        auto possible_steals = std::make_optional<std::vector<std::pair<AbsolutePos, std::vector<Open>>>>();
        auto position = AbsolutePos((ToUType(latest_discarder_) + 1) % 4);
        auto discarded_tile = player(latest_discarder_).latest_discard();
        while (position != latest_discarder_) {
            auto stealer = AbsolutePos(position);
            auto possible_opens = player(stealer).PossibleOpensAfterOthersDiscard(discarded_tile, ToRelativePos(position, latest_discarder_));
            possible_steals->emplace_back(std::make_pair(stealer, std::move(possible_opens)));
            position = AbsolutePos((ToUType(position) + 1) % 4);
        }
        return possible_steals;
    }

    RelativePos State::ToRelativePos(AbsolutePos origin, AbsolutePos target) {
        assert(origin != target);
        switch ((ToUType(target) - ToUType(origin) + 4) % 4) {
            case 1:
                return RelativePos::kRight;
            case 2:
                return RelativePos::kMid;
            case 3:
                return RelativePos::kLeft;
        }
        assert(false);
    }

    State::State(const std::string &json_str): State() {
        std::unique_ptr<mjproto::State> state = std::make_unique<mjproto::State>();
        auto status = google::protobuf::util::JsonStringToMessage(json_str, state.get());
        assert(status.ok());

        for (int i = 0; i < 4; ++i) player_ids_[i] = state->player_ids(i);
        // Set scores
        score_ = Score(state->init_score());
        // Set walls
        auto wall_tiles = std::vector<Tile>();
        for (auto tile_id: state->wall()) wall_tiles.emplace_back(Tile(tile_id));
        wall_ = Wall(score_.round(), wall_tiles);
        // Set initial hands
        for (int i = 0; i < 4; ++i) players_[i] = Player{AbsolutePos(i), River(),
                                                         Hand(wall_.initial_hand_tiles(AbsolutePos(i)))};
    }

    std::string State::ToJson() const {
        std::string serialized;
        std::unique_ptr<mjproto::State> state = std::make_unique<mjproto::State>();
        // Set scores
        state->mutable_init_score()->set_round(score_.round());
        state->mutable_init_score()->set_honba(score_.honba());
        state->mutable_init_score()->set_riichi(score_.riichi());
        for (int i = 0; i < 4; ++i) state->mutable_init_score()->mutable_ten()->Add(score_.ten()[i]);
        // Set walls
        for(auto t: wall_.tiles())state->mutable_wall()->Add(t.Id());
        // Set initial hands
        for(int i = 0; i < 4; ++i) {
            state->add_init_hands();
            for (auto t: wall_.initial_hand_tiles(AbsolutePos(i))) {
                state->mutable_init_hands(i)->add_tiles(t.Id());
            }
        }

        auto status = google::protobuf::util::MessageToJsonString(*state, &serialized);
        assert(status.ok());
        return serialized;
    }
}  // namespace mj
