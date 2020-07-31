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

        event_history_ = mjproto::EventHistory();
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
        auto observation = Observation(pos, score_, mutable_player(pos));
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
        // Set init hands
        for (int i = 0; i < 4; ++i) {
            players_[i] = Player{AbsolutePos(i), River(), Hand(wall_.initial_hand_tiles(AbsolutePos(i)))};
            for (auto t: wall_.initial_hand_tiles(AbsolutePos(i))) private_infos_[i].add_init_hand(t.Id());
        }
        // Set event history
        std::vector<int> draw_ixs = {0, 0, 0, 0};
        for (int i = 0; i < state->event_history().events_size(); ++i) {
            auto event = state->event_history().events(i);
            auto who = AbsolutePos(event.who());
            switch (event.type()) {
                case mjproto::EVENT_TYPE_DRAW:
                    // TODO: wrap by func
                    private_infos_[ToUType(who)].add_draws(state->private_infos(ToUType(who)).draws(draw_ixs[ToUType(who)]));
                    draw_ixs[ToUType(who)]++;
                    break;
                case mjproto::EVENT_TYPE_DISCARD_FROM_HAND:
                case mjproto::EVENT_TYPE_DISCARD_DRAWN_TILE:
                    break;
                case mjproto::EVENT_TYPE_RIICHI:
                    break;
                case mjproto::EVENT_TYPE_TSUMO:
                    break;
                case mjproto::EVENT_TYPE_RON:
                case mjproto::EVENT_TYPE_CHI:
                case mjproto::EVENT_TYPE_PON:
                case mjproto::EVENT_TYPE_KAN_CLOSED:
                case mjproto::EVENT_TYPE_KAN_OPENED:
                case mjproto::EVENT_TYPE_KAN_ADDED:
                    break;
                case mjproto::EVENT_TYPE_NEW_DORA:
                    break;
                case mjproto::EVENT_TYPE_RIICHI_SCORE_CHANGE:
                    break;
            }
            event_history_.add_events();
            event_history_.mutable_events(i)->set_who(event.who());
            event_history_.mutable_events(i)->set_type(event.type());
            event_history_.mutable_events(i)->set_tile(event.tile());
            event_history_.mutable_events(i)->set_open(event.open());
        }
    }

    std::string State::ToJson() const {
        std::string serialized;
        std::unique_ptr<mjproto::State> state = std::make_unique<mjproto::State>();
        // Set player ids
        for (const auto &player_id: player_ids_) state->add_player_ids(player_id);
        // Set scores
        state->mutable_init_score()->set_round(score_.round());
        state->mutable_init_score()->set_honba(score_.honba());
        state->mutable_init_score()->set_riichi(score_.riichi());
        for (int i = 0; i < 4; ++i) state->mutable_init_score()->mutable_ten()->Add(score_.ten()[i]);
        // Set walls
        for(auto t: wall_.tiles())state->mutable_wall()->Add(t.Id());
        // Set doras and ura doras
        for (auto dora: wall_.doras()) state->add_doras(dora.Id());
        for (auto ura_dora: wall_.ura_doras()) state->add_ura_doras(ura_dora.Id());
        // Set init hands
        for(int i = 0; i < 4; ++i) {
            state->add_private_infos();
            state->mutable_private_infos(i)->set_who(mjproto::AbsolutePos(i));
            for (auto t: private_infos_[i].init_hand()) state->mutable_private_infos(i)->add_init_hand(t);
        }
        // Set draws
        for (int i = 0; i < 4; ++i) {
            for (auto draw: private_infos_[i].draws()) {
                state->mutable_private_infos(i)->add_draws(draw);
            }
        }
        // Set event history
        for (int i = 0; i < event_history_.events_size(); ++i) {
            auto event = event_history_.events(i);
            state->mutable_event_history()->add_events();
            state->mutable_event_history()->mutable_events(i)->set_who(event.who());
            state->mutable_event_history()->mutable_events(i)->set_type(event.type());
            state->mutable_event_history()->mutable_events(i)->set_tile(event.tile());
            state->mutable_event_history()->mutable_events(i)->set_open(event.open());
        }

        auto status = google::protobuf::util::MessageToJsonString(*state, &serialized);
        assert(status.ok());
        return serialized;
    }

    std::pair<AbsolutePos, Tile> State::Draw() {
        auto draw = wall_.Draw();
        mutable_player(drawer_).Draw(draw);
        private_infos_[ToUType(drawer_)].add_draws(draw.Id());
        return {drawer_, draw};
    }
}  // namespace mj
