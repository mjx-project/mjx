#include "state.h"
#include "utils.h"

#include <google/protobuf/util/json_util.h>

namespace mj
{
    State::State(std::uint32_t seed)
    : seed_(seed), wall_(0)
    {
        // TODO (sotetsuk): shuffle seats
    }

    void State::InitRound() {
        // TODO: use seed_
        last_event_ = EventType::kDiscardDrawnTile;
        drawer_ = dealer();
        latest_discarder_ = AbsolutePos::kInitNorth;
        wall_ = Wall(round());  // TODO: use seed_
        for (int i = 0; i < 4; ++i) players_[i] = Player{AbsolutePos(i), River(), Hand(wall_.initial_hand_tiles(AbsolutePos(i)))};
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

    Player& State::mutable_player(AbsolutePos pos) {
        return players_.at(ToUType(pos));
    }

    const Player &State::player(AbsolutePos pos) const {
        return players_.at(ToUType(pos));
    }

    Observation State::CreateObservation(AbsolutePos who) {
        auto observation = Observation(who, state_);  // TODO: fix
        switch (last_event_) {
            case EventType::kDraw:
                observation.add_possible_action(PossibleAction::CreateDiscard(player(who).PossibleDiscards()));
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
        auto discarded_tile = mutable_player(latest_discarder_).latest_discard();
        while (position != latest_discarder_) {
            if (mutable_player(position).CanRon(discarded_tile)) possible_winners->emplace_back(position);
            position = AbsolutePos((ToUType(position) + 1) % 4);
        }
        if (possible_winners.value().empty()) possible_winners = std::nullopt;
        return possible_winners;
    }

    std::optional<std::vector<std::pair<AbsolutePos, std::vector<Open>>>> State::StealCheck() {
        assert(Any(last_event_, { EventType::kDiscardFromHand, EventType::kDiscardDrawnTile }));
        auto possible_steals = std::make_optional<std::vector<std::pair<AbsolutePos, std::vector<Open>>>>();
        auto position = AbsolutePos((ToUType(latest_discarder_) + 1) % 4);
        auto discarded_tile = mutable_player(latest_discarder_).latest_discard();
        while (position != latest_discarder_) {
            auto stealer = AbsolutePos(position);
            auto possible_opens = mutable_player(stealer).PossibleOpensAfterOthersDiscard(discarded_tile, ToRelativePos(position, latest_discarder_));
            possible_steals->emplace_back(std::make_pair(stealer, std::move(possible_opens)));
            position = AbsolutePos((ToUType(position) + 1) % 4);
        }
        return possible_steals;
    }

    RelativePos State::ToRelativePos(AbsolutePos origin, AbsolutePos target) {
        switch ((ToUType(target) - ToUType(origin) + 4) % 4) {
            case 0:
                return RelativePos::kSelf;
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
        state_.mutable_init_score()->CopyFrom(state->init_score());
        state_.mutable_terminal()->mutable_final_score()->CopyFrom(state->init_score());
        // Set walls
        auto wall_tiles = std::vector<Tile>();
        for (auto tile_id: state->wall()) wall_tiles.emplace_back(Tile(tile_id));
        wall_ = Wall(round(), wall_tiles);
        // Set init hands
        for (int i = 0; i < 4; ++i) {
            players_[i] = Player{AbsolutePos(i), River(), Hand(wall_.initial_hand_tiles(AbsolutePos(i)))};
            state_.mutable_private_infos()->Add();
            state_.mutable_private_infos(i)->set_who(mjproto::AbsolutePos(i));
            for (auto t: wall_.initial_hand_tiles(AbsolutePos(i))) {
                state_.mutable_private_infos(i)->add_init_hand(t.Id());
            }
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
                    Tsumo(who);
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
        state->mutable_init_score()->CopyFrom(state_.init_score());
        // Set walls
        for(auto t: wall_.tiles())state->mutable_wall()->Add(t.Id());
        // Set doras and ura doras
        for (auto dora: wall_.dora_indicators()) state->add_doras(dora.Id());
        for (auto ura_dora: wall_.ura_dora_indicators()) state->add_ura_doras(ura_dora.Id());
        // Set private infos
        state->mutable_private_infos()->CopyFrom(state_.private_infos());
        // Set event history
        state->mutable_event_history()->CopyFrom(state_.event_history());
        // Set terminal
        state->mutable_terminal()->CopyFrom(state_.terminal());

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
        state_.mutable_event_history()->mutable_events()->Add(std::move(event));
        state_.mutable_private_infos(ToUType(who))->add_draws(draw.Id());

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
        state_.mutable_event_history()->mutable_events()->Add(std::move(event));
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
        state_.mutable_event_history()->mutable_events()->Add(std::move(event));

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
        state_.mutable_event_history()->mutable_events()->Add(std::move(event));

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
        auto doras = wall_.dora_indicators();
        event.set_tile(doras.back().Id());
        state_.mutable_event_history()->mutable_events()->Add(std::move(event));

        // set last action
        last_event_ = EventType::kNewDora;
    }

    void State::RiichiScoreChange() {
        state_.mutable_terminal()->mutable_final_score()->set_riichi(riichi() + 1);
        state_.mutable_terminal()->mutable_final_score()->set_ten(ToUType(last_action_taker_), ten(last_action_taker_) - 1000);

        // set proto
        mjproto::Event event{};
        event.set_who(mjproto::AbsolutePos(last_action_taker_));
        event.set_type(mjproto::EVENT_TYPE_RIICHI_SCORE_CHANGE);
        state_.mutable_event_history()->mutable_events()->Add(std::move(event));

        // set last action
        last_event_ = EventType::kRiichiScoreChange;
    }

    void State::Tsumo(AbsolutePos winner) {
        mutable_player(winner).Tsumo();
        auto [hand_info, win_score] = EvalWinHand(winner);
        // calc ten moves
        auto [ten_, ten_moves] = win_score.TenMoves(winner, dealer());
        for (auto &[who, ten_move]: ten_moves) {
            if (ten_move > 0) ten_move += riichi() * 1000 + honba() * 300;
            else if (ten_move < 0) ten_move -= honba() * 100;
        }

        // set event
        mjproto::Event event{};
        event.set_who(mjproto::AbsolutePos(winner));
        event.set_type(mjproto::EVENT_TYPE_TSUMO);
        assert(hand_info.win_tile);
        event.set_tile(hand_info.win_tile.value().Id());
        state_.mutable_event_history()->mutable_events()->Add(std::move(event));

        // set terminal
        mjproto::Win win;
        win.set_who(mjproto::AbsolutePos(winner));
        win.set_from_who(mjproto::AbsolutePos(winner));
        // winner closed tiles, opens and win tile
        for (auto t: hand_info.closed_tiles) {
            win.add_closed_tiles(t.Id());
        }
        for (const auto &open: hand_info.opens) {
            win.add_opens(open.GetBits());
        }
        assert(hand_info.win_tile);
        win.set_win_tile(hand_info.win_tile.value().Id());
        // fu
        if (win_score.fu()) win.set_fu(win_score.fu().value());
        // yaku, fans
        for (const auto &[yaku, fan]: win_score.yaku()) {
            if (yaku == Yaku::kReversedDora) continue;  // mjlog puts ura-dora at last
            win.add_yakus(ToUType(yaku));
            win.add_fans(fan);
        }
        if (auto has_ura_dora = win_score.HasYaku(Yaku::kReversedDora); has_ura_dora) {
            win.add_yakus(ToUType(Yaku::kReversedDora));
            win.add_fans(has_ura_dora.value());
        }
        // ten and ten moves
        win.set_ten(ten_);
        for (int i = 0; i < 4; ++i) win.add_ten_changes(0);
        for (const auto &[who, ten_move]: ten_moves) {
            win.set_ten_changes(ToUType(who), ten_move);
            state_.mutable_terminal()->mutable_final_score()->set_ten(ToUType(who), ten(who) + ten_move);
        }
        state_.mutable_terminal()->mutable_wins()->Add(std::move(win));
        state_.mutable_terminal()->set_is_game_over(IsGameOver());

        // set last action
        last_action_taker_ = winner;
        last_event_ = EventType::kTsumo;
    }

    void State::Ron(AbsolutePos winner, AbsolutePos loser, Tile tile) {
        mutable_player(winner).Ron(tile);
        auto [hand_info, win_score] = EvalWinHand(winner);
        // calc ten moves
        auto [ten_, ten_moves] = win_score.TenMoves(winner, dealer(), loser);
        for (auto &[who, ten_move]: ten_moves) {
            if (ten_move > 0) ten_move += riichi() * 1000 + honba() * 300;
            else if (ten_move < 0) ten_move -= honba() * 300;
        }

        // set event
        mjproto::Event event{};
        event.set_who(mjproto::AbsolutePos(winner));
        event.set_type(mjproto::EVENT_TYPE_RON);
        event.set_tile(tile.Id());
        state_.mutable_event_history()->mutable_events()->Add(std::move(event));

        // set terminal
        mjproto::Win win;
        win.set_who(mjproto::AbsolutePos(winner));
        win.set_from_who(mjproto::AbsolutePos(loser));
        // winner closed tiles, opens and win tile
        for (auto t: hand_info.closed_tiles) {
            win.add_closed_tiles(t.Id());
        }
        for (const auto &open: hand_info.opens) {
            win.add_opens(open.GetBits());
        }
        win.set_win_tile(tile.Id());
        // fu
        if (win_score.fu()) win.set_fu(win_score.fu().value());
        // yaku, fans
        for (const auto &[yaku, fan]: win_score.yaku()) {
            if (yaku == Yaku::kReversedDora) continue;  // mjlog puts ura-dora at last
            win.add_yakus(ToUType(yaku));
            win.add_fans(fan);
        }
        if (auto has_ura_dora = win_score.HasYaku(Yaku::kReversedDora); has_ura_dora) {
            win.add_yakus(ToUType(Yaku::kReversedDora));
            win.add_fans(has_ura_dora.value());
        }
        // ten and ten moves
        win.set_ten(ten_);
        for (int i = 0; i < 4; ++i) win.add_ten_changes(0);
        for (const auto &[who, ten_move]: ten_moves) {
            win.set_ten_changes(ToUType(who), ten_move);
            state_.mutable_terminal()->mutable_final_score()->set_ten(ToUType(who), ten(who) + ten_move);
        }
        // set win to terminal
        state_.mutable_terminal()->mutable_wins()->Add(std::move(win));
        state_.mutable_terminal()->set_is_game_over(IsGameOver());

        // set last action
        last_action_taker_ = winner;
        last_event_ = EventType::kRon;
    }

    void State::NoWinner() {
        // set event
        mjproto::Event event{};
        event.set_type(mjproto::EVENT_TYPE_NO_WINNER);
        state_.mutable_event_history()->mutable_events()->Add(std::move(event));

        // set terminal
        std::vector<int> is_tenpai = {0, 0, 0, 0};
        for (int i = 0; i < 4; ++i) {
            auto who = AbsolutePos(i);
            if (auto tenpai_hand = player(who).EvalTenpai(); tenpai_hand) {
                is_tenpai[i] = 1;
                mjproto::TenpaiHand tenpai;
                tenpai.set_who(mjproto::AbsolutePos(who));
                for (auto tile: tenpai_hand.value().closed_tiles) {
                    tenpai.mutable_closed_tiles()->Add(tile.Id());
                }
                state_.mutable_terminal()->mutable_no_winner()->mutable_tenpais()->Add(std::move(tenpai));
            }
        }
        auto num_tenpai = std::accumulate(is_tenpai.begin(), is_tenpai.end(), 0);
        for (int i = 0; i < 4; ++i) {
            int ten_move;
            switch (num_tenpai) {
                case 1:
                    ten_move = is_tenpai[i] ? 3000 : -1000;
                    break;
                case 2:
                    ten_move = is_tenpai[i] ? 1500 : -1500;
                    break;
                case 3:
                    ten_move = is_tenpai[i] ? 1000 : -3000;
                    break;
                default:  // 0, 4
                    ten_move = 0;
                    break;
            }
            // apply ten moves
            state_.mutable_terminal()->mutable_no_winner()->add_ten_changes(ten_move);
            state_.mutable_terminal()->mutable_final_score()->set_ten(i, ten(AbsolutePos(i)) + ten_move);
        }
        state_.mutable_terminal()->set_is_game_over(IsGameOver());

        // set last action
        last_event_ = EventType::kNoWinner;
    }

    bool State::IsGameOver() {
        // TODO (sotetsuk): 西入後の終曲条件が供託未収と書いてあるので、修正が必要。　https://tenhou.net/man/
        // ラス親のあがりやめも考慮しないといけない
        auto tens_ = tens();
        bool is_game_over = *std::min_element(tens_.begin(), tens_.end()) < 0 ||
                (round() >= 7 && *std::max_element(tens_.begin(), tens_.end()) >= 30000);
        return is_game_over;
    }

    std::pair<HandInfo, WinScore> State::EvalWinHand(AbsolutePos who) const noexcept {
        // TODO: 場風, 自風, 海底, 一発, 両立直, 天和・地和, 親・子, ドラ, 裏ドラ の情報を追加する
        auto seat_wind = ToSeatWind(who, dealer());
        auto win_state_info = WinStateInfo(
                seat_wind,
                prevalent_wind(),
                false,
                false,
                false,
                false,
                seat_wind == Wind::kEast,
                wall_.dora_count(),
                wall_.ura_dora_count());
        return player(who).EvalWinHand(std::move(win_state_info));
    }

    Wind State::ToSeatWind(AbsolutePos who, AbsolutePos dealer) {
        return Wind((ToUType(who) - ToUType(dealer) + 4) % 4);
    }

    AbsolutePos State::dealer() const {
        return AbsolutePos(state_.init_score().round() % 4);
    }

    std::uint8_t State::round() const {
        return state_.terminal().final_score().round();
    }

    std::uint8_t State::honba() const {
        return state_.terminal().final_score().honba();
    }

    std::uint8_t State::riichi() const {
        return state_.terminal().final_score().riichi();
    }

    std::array<std::int32_t, 4> State::tens() const {
        std::array<std::int32_t, 4> tens{};
        for (int i = 0; i < 4; ++i) tens[i] = state_.terminal().final_score().ten(i);
        return tens;
    }

    Wind State::prevalent_wind() const {
        return Wind(round() / 4);
    }

    std::int32_t State::ten(AbsolutePos who) const {
        return state_.terminal().final_score().ten(ToUType(who));
    }
}  // namespace mj
