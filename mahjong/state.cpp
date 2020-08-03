#include "state.h"
#include "utils.h"

#include <google/protobuf/util/json_util.h>

namespace mj
{
    State::State(std::uint32_t seed)
    : seed_(seed), curr_score_(Score()), wall_(0)
    {
        // TODO (sotetsuk): shuffle seats
    }

    void State::InitRound() {
        // TODO: use seed_
        last_event_ = EventType::kDiscardDrawnTile;
        dealer_ = AbsolutePos(curr_score_.round() % 4);
        drawer_ = dealer_;
        latest_discarder_ = AbsolutePos::kInitNorth;
        wall_ = Wall(curr_score_.round());  // TODO: use seed_
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
        mutable_player(drawer_).Draw(wall_.Draw());
        // TODO (sotetsuk): update action history
        last_event_ = EventType::kDraw;
        return drawer_;
    }

    void State::UpdateStateByAction(const Action &action) {
        switch (action.type()) {
            case ActionType::kDiscard:
                auto [tile, tsumogiri] = mutable_player(action.who()).Discard(action.discard());
                last_event_ = tsumogiri ? EventType::kDiscardDrawnTile : EventType::kDiscardFromHand;
                drawer_ = AbsolutePos((static_cast<int>(action.who()) + 1) % 4);
                latest_discarder_ = action.who();
                break;
        }
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
        auto observation = Observation(pos, curr_score_, mutable_player(pos));
        switch (last_event_) {
            case EventType::kDraw:
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
        assert(Any(last_event_, { EventType::kDiscardFromHand, EventType::kDiscardDrawnTile }));
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
        init_score_ = Score(state->init_score());
        curr_score_ = Score(state->init_score());
        // Set walls
        auto wall_tiles = std::vector<Tile>();
        for (auto tile_id: state->wall()) wall_tiles.emplace_back(Tile(tile_id));
        wall_ = Wall(curr_score_.round(), wall_tiles);
        // Set init hands
        for (int i = 0; i < 4; ++i) {
            players_[i] = Player{AbsolutePos(i), River(), Hand(wall_.initial_hand_tiles(AbsolutePos(i)))};
            for (auto t: wall_.initial_hand_tiles(AbsolutePos(i))) private_infos_[i].add_init_hand(t.Id());
        }
        // Set event history
        std::vector<int> draw_ixs = {0, 0, 0, 0};
        AbsolutePos last_discarder;
        for (int i = 0; i < state->event_history().events_size(); ++i) {
            auto event = state->event_history().events(i);
            auto who = AbsolutePos(event.who());
            switch (event.type()) {
                case mjproto::EVENT_TYPE_DRAW:
                    // TODO: wrap by func
                    // private_infos_[ToUType(who)].add_draws(state->private_infos(ToUType(who)).draws(draw_ixs[ToUType(who)]));
                    // draw_ixs[ToUType(who)]++;
                    Draw(who);
                    break;
                case mjproto::EVENT_TYPE_DISCARD_FROM_HAND:
                case mjproto::EVENT_TYPE_DISCARD_DRAWN_TILE:
                    Discard(who, Tile(event.tile()));
                    last_discarder = who;
                    break;
                case mjproto::EVENT_TYPE_RIICHI:
                    Riichi(who);
                    break;
                case mjproto::EVENT_TYPE_TSUMO:
                    Tsumo(who, Tile(event.tile()));
                    break;
                case mjproto::EVENT_TYPE_RON:
                    Ron(who, last_discarder, Tile(event.tile()));
                    break;
                case mjproto::EVENT_TYPE_CHI:
                case mjproto::EVENT_TYPE_PON:
                case mjproto::EVENT_TYPE_KAN_CLOSED:
                case mjproto::EVENT_TYPE_KAN_OPENED:
                case mjproto::EVENT_TYPE_KAN_ADDED:
                    ApplyOpen(who, Open(event.open()));
                    break;
                case mjproto::EVENT_TYPE_NEW_DORA:
                    AddNewDora();
                    break;
                case mjproto::EVENT_TYPE_RIICHI_SCORE_CHANGE:
                    RiichiScoreChange();
                    break;
                case mjproto::EVENT_TYPE_NO_WINNER:
                    NoWinner();
                    break;
            }
        }
    }

    std::string State::ToJson() const {
        std::string serialized;
        std::unique_ptr<mjproto::State> state = std::make_unique<mjproto::State>();
        // Set player ids
        for (const auto &player_id: player_ids_) state->add_player_ids(player_id);
        // Set scores
        state->mutable_init_score()->set_round(init_score_.round());
        state->mutable_init_score()->set_honba(init_score_.honba());
        state->mutable_init_score()->set_riichi(init_score_.riichi());
        for (int i = 0; i < 4; ++i) state->mutable_init_score()->mutable_ten()->Add(init_score_.ten()[i]);
        // Set walls
        for(auto t: wall_.tiles())state->mutable_wall()->Add(t.Id());
        // Set doras and ura doras
        for (auto dora: wall_.doras()) state->add_doras(dora.Id());
        for (auto ura_dora: wall_.ura_doras()) state->add_ura_doras(ura_dora.Id());
        // Set private infos
        for(int i = 0; i < 4; ++i) {
            state->add_private_infos();
            state->mutable_private_infos(i)->CopyFrom(private_infos_[i]);
            state->mutable_private_infos(i)->set_who(mjproto::AbsolutePos(i));
        }
        // Set event history
        state->mutable_event_history()->CopyFrom(event_history_);
        // Set terminal
        state->mutable_terminal()->CopyFrom(terminal_);

        auto status = google::protobuf::util::MessageToJsonString(*state, &serialized);
        assert(status.ok());
        return serialized;
    }

    Tile State::Draw(AbsolutePos who) {
        bool is_kan_draw = last_action_taker_ == who && Any(last_event_, {EventType::kKanClosed, EventType::kKanOpened, EventType::kKanAdded});
        auto draw = is_kan_draw ? wall_.KanDraw() : wall_.Draw();
        mutable_player(who).Draw(draw);

        // set proto
        mjproto::Event event{};
        event.set_who(mjproto::AbsolutePos(who));
        event.set_type(mjproto::EVENT_TYPE_DRAW);
        event_history_.mutable_events()->Add(std::move(event));
        private_infos_[ToUType(who)].add_draws(draw.Id());

        // set last action
        last_action_taker_ = who;
        last_event_ = EventType::kDraw;

        return draw;
    }

    void State::Discard(AbsolutePos who, Tile discard) {
        auto [discarded, tsumogiri] = mutable_player(who).Discard(discard);
        assert(discard == discarded);

        // set proto
        mjproto::Event event{};
        event.set_who(mjproto::AbsolutePos(who));
        event.set_type(tsumogiri ? mjproto::EVENT_TYPE_DISCARD_DRAWN_TILE : mjproto::EVENT_TYPE_DISCARD_FROM_HAND);
        event.set_tile(discard.Id());
        event_history_.mutable_events()->Add(std::move(event));
        // TODO: set discarded tile to river

        // set last action
        last_action_taker_ = who;
        last_event_ = tsumogiri ? EventType::kDiscardDrawnTile : EventType::kDiscardFromHand;
    }

    void State::Riichi(AbsolutePos who) {
        mutable_player(who).Riichi();

        // set proto
        mjproto::Event event{};
        event.set_who(mjproto::AbsolutePos(who));
        event.set_type(mjproto::EVENT_TYPE_RIICHI);
        event_history_.mutable_events()->Add(std::move(event));

        // set last action
        last_action_taker_ = who;
        last_event_ = EventType::kRiichi;
    }

    void State::ApplyOpen(AbsolutePos who, Open open) {
        mutable_player(who).ApplyOpen(open);

        // set proto
        mjproto::Event event{};
        event.set_who(mjproto::AbsolutePos(who));
        auto open_type = open.Type();
        auto to_event_type = [](OpenType open_type) {
            switch (open_type) {
                case OpenType::kChi:
                    return EventType::kChi;
                case OpenType::kPon:
                    return EventType::kPon;
                case OpenType::kKanOpened:
                    return EventType::kKanOpened;
                case OpenType::kKanClosed:
                    return EventType::kKanClosed;
                case OpenType::kKanAdded:
                    return EventType::kKanAdded;
            }
        };
        event.set_type(mjproto::EventType(to_event_type(open_type)));
        event.set_open(open.GetBits());
        event_history_.mutable_events()->Add(std::move(event));

        // set last action
        last_action_taker_ = who;
        switch (open_type) {
            case OpenType::kChi:
                last_event_ = EventType::kChi;
                break;
            case OpenType::kPon:
                last_event_ = EventType::kPon;
                break;
            case OpenType::kKanOpened:
                last_event_ = EventType::kKanOpened;
                break;
            case OpenType::kKanClosed:
                last_event_ = EventType::kKanClosed;
                break;
            case OpenType::kKanAdded:
                last_event_ = EventType::kKanAdded;
                break;
        }
    }

    void State::AddNewDora() {
        wall_.AddKanDora();

        // set proto
        mjproto::Event event{};
        event.set_type(mjproto::EVENT_TYPE_NEW_DORA);
        auto doras = wall_.doras();
        event.set_tile(doras.back().Id());
        event_history_.mutable_events()->Add(std::move(event));

        // set last action
        last_event_ = EventType::kNewDora;
    }

    void State::RiichiScoreChange() {
        curr_score_.Riichi(last_action_taker_);

        // set proto
        mjproto::Event event{};
        event.set_who(mjproto::AbsolutePos(last_action_taker_));
        event.set_type(mjproto::EVENT_TYPE_RIICHI_SCORE_CHANGE);
        event_history_.mutable_events()->Add(std::move(event));

        // set last action
        last_event_ = EventType::kRiichiScoreChange;
    }

    void State::Tsumo(AbsolutePos who, Tile tile) {
        // set event
        mjproto::Event event{};
        event.set_who(mjproto::AbsolutePos(who));
        event.set_type(mjproto::EVENT_TYPE_TSUMO);
        event.set_tile(tile.Id());
        event_history_.mutable_events()->Add(std::move(event));

        // set terminal

        // set last action
        last_action_taker_ = who;
        last_event_ = EventType::kTsumo;
    }

    void State::Ron(AbsolutePos who, AbsolutePos from_who, Tile tile) {
        // set event
        mjproto::Event event{};
        event.set_who(mjproto::AbsolutePos(who));
        event.set_type(mjproto::EVENT_TYPE_RON);
        event.set_tile(tile.Id());
        event_history_.mutable_events()->Add(std::move(event));

        // set terminal

        // set last action
        last_action_taker_ = who;
        last_event_ = EventType::kRon;
    }

    void State::NoWinner() {
        // set event
        mjproto::Event event{};
        event.set_type(mjproto::EVENT_TYPE_NO_WINNER);
        event_history_.mutable_events()->Add(std::move(event));

        // set terminal
        std::vector<int> is_tenpai = {0, 0, 0, 0};
        for (int i = 0; i < 4; ++i) {
            auto who = AbsolutePos(i);
            if (hand(who).IsTenpai()) {
                is_tenpai[i] = 1;
                mjproto::TenpaiHand tenpai;
                tenpai.set_who(mjproto::AbsolutePos(who));
                for (auto tile: hand(who).ToVectorClosed(true)) {
                    tenpai.mutable_closed_tiles()->Add(tile.Id());
                }
                terminal_.mutable_no_winner()->mutable_tenpais()->Add(std::move(tenpai));
            }
        }
        auto num_tenpai = std::accumulate(is_tenpai.begin(), is_tenpai.end(), 0);
        for (int i = 0; i < 4; ++i) {
            int ten;
            switch (num_tenpai) {
                case 1:
                    ten = is_tenpai[i] ? 3000 : -1000;
                    break;
                case 2:
                    ten = is_tenpai[i] ? 1500 : -1500;
                    break;
                case 3:
                    ten = is_tenpai[i] ? 1000 : -3000;
                    break;
                default:  // 0, 4
                    ten = 0;
                    break;
            }
            terminal_.mutable_no_winner()->add_ten_changes(ten);
        }

        // set last action
        last_event_ = EventType::kNoWinner;
    }
}  // namespace mj
