#include "state.h"
#include "utils.h"

#include <google/protobuf/util/json_util.h>

namespace mj
{
    State::State(std::vector<PlayerId> player_ids, std::uint32_t seed, int round, int honba, int riichi, std::array<int, 4> tens)
    : seed_(seed), wall_(0, seed) {
        // TODO: use seed_
        assert(std::set(player_ids.begin(), player_ids.end()).size() == 4);  // player_ids should be identical
        last_event_ = Event();
        for (int i = 0; i < 4; ++i)
            players_[i] = Player{player_ids[i], AbsolutePos(i), River(), Hand(wall_.initial_hand_tiles(AbsolutePos(i)))};

        // set protos
        // player_ids
        for (int i = 0; i < 4; ++i) state_.add_player_ids(player_ids[i]);
        // init_score
        state_.mutable_init_score()->set_round(round);
        state_.mutable_init_score()->set_honba(honba);
        state_.mutable_init_score()->set_riichi(riichi);
        for (int i = 0; i < 4; ++i) state_.mutable_init_score()->add_ten(tens[i]);
        curr_score_.CopyFrom(state_.init_score());
        // wall
        for(auto t: wall_.tiles()) state_.mutable_wall()->Add(t.Id());
        // doras, ura_doras
        state_.add_doras(wall_.dora_indicators().front().Id());
        state_.add_ura_doras(wall_.ura_dora_indicators().front().Id());
        // private info
        for (int i = 0; i < 4; ++i) state_.add_private_infos()->set_who(mjproto::AbsolutePos(i));
    }

    bool State::IsRoundOver() const {
        if (!wall_.HasDrawLeft()) return true;
        return false;
    }

    Player& State::mutable_player(AbsolutePos pos) {
        return players_.at(ToUType(pos));
    }

    const Player &State::player(AbsolutePos pos) const {
        return players_.at(ToUType(pos));
    }

    std::unordered_map<PlayerId, Observation> State::CreateObservations() const {
        switch (last_event_.type()) {
            case EventType::kDraw:
            case EventType::kNewDora:
                {
                    auto who = last_draw_.who();
                    auto player_id = player(who).player_id();
                    auto observation = Observation(who, state_);

                    // => Tsumo (1)
                    if (player(who).IsCompleted() && player(who).CanTsumo(win_state_info(who)))
                        observation.add_possible_action(PossibleAction::CreateTsumo());

                    // => Kan (2)
                    if (auto possible_kans = player(who).PossibleOpensAfterDraw(); !possible_kans.empty())
                        observation.add_possible_action(PossibleAction::CreateKanAdded());

                    // => Riichi (3)
                    if (player(who).CanRiichi())
                        observation.add_possible_action(PossibleAction::CreateRiichi());

                    // => Discard (4)
                    observation.add_possible_action(PossibleAction::CreateDiscard(player(who).PossibleDiscards()));

                    return { {player_id, std::move(observation)} };
                }
            case EventType::kRiichi:
                {
                    // => Discard (5)
                    auto who = last_draw_.who();
                    auto observation = Observation(who, state_);
                    observation.add_possible_action(PossibleAction::CreateDiscard(player(who).PossibleDiscardsAfterRiichi()));
                    return { {player(who).player_id(), std::move(observation)} };
                }
            case EventType::kChi:
            case EventType::kPon:
                {
                    // => Discard (6)
                    auto who = last_event_.who();
                    auto observation = Observation(who, state_);
                    observation.add_possible_action(PossibleAction::CreateDiscard(player(who).PossibleDiscards()));
                    return { {player(who).player_id(), std::move(observation)} };
                }
            case EventType::kDiscardFromHand:
            case EventType::kDiscardDrawnTile:
            case EventType::kRiichiScoreChange:  // TODO: RiichiScoreChange => Ron はありえる？
                // => Ron (7)
                // => Chi, Pon and KanOpened (8)
                assert(last_action_.type() != ActionType::kNo);
                if (auto steal_observations = CreateStealAndRonObservation(); !steal_observations.empty()) return steal_observations;
            case EventType::kTsumo:
            case EventType::kRon:
            case EventType::kKanClosed:
            case EventType::kKanOpened:
            case EventType::kKanAdded:
            case EventType::kNoWinner:
                assert(false);
        }
        assert(false);
    }

    State::State(const std::string &json_str) {
         std::unique_ptr<mjproto::State> state = std::make_unique<mjproto::State>();
         auto status = google::protobuf::util::JsonStringToMessage(json_str, state.get());
         assert(status.ok());

         // Set player ids
         state_.mutable_player_ids()->CopyFrom(state->player_ids());
         // Set scores
         state_.mutable_init_score()->CopyFrom(state->init_score());
         curr_score_.CopyFrom(state->init_score());
         // Set walls
         auto wall_tiles = std::vector<Tile>();
         for (auto tile_id: state->wall()) wall_tiles.emplace_back(Tile(tile_id));
         wall_ = Wall(round(), wall_tiles);
         state_.mutable_wall()->CopyFrom(state->wall());
         // Set dora
         state_.add_doras(wall_.dora_indicators().front().Id());
         state_.add_ura_doras(wall_.ura_dora_indicators().front().Id());
         // Set init hands
         for (int i = 0; i < 4; ++i) {
             players_[i] = Player{state_.player_ids(i), AbsolutePos(i), River(), Hand(wall_.initial_hand_tiles(AbsolutePos(i)))};
             state_.mutable_private_infos()->Add();
             state_.mutable_private_infos(i)->set_who(mjproto::AbsolutePos(i));
             for (auto t: wall_.initial_hand_tiles(AbsolutePos(i))) {
                 state_.mutable_private_infos(i)->add_init_hand(t.Id());
             }
         }
         // Set event history
         std::vector<int> draw_ixs = {0, 0, 0, 0};
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
                     break;
                 case mjproto::EVENT_TYPE_RIICHI:
                     Riichi(who);
                     break;
                 case mjproto::EVENT_TYPE_TSUMO:
                     Tsumo(who);
                     break;
                 case mjproto::EVENT_TYPE_RON:
                     assert(last_discard_.tile() == Tile(event.tile()));
                     Ron(who, last_discard_.who(), Tile(event.tile()));
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
        auto status = google::protobuf::util::MessageToJsonString(state_, &serialized);
        assert(status.ok());
        return serialized;
    }

    Tile State::Draw(AbsolutePos who) {
        bool is_kan_draw = last_event_.who() == who && Any(last_event_.type(), {EventType::kKanClosed, EventType::kKanOpened, EventType::kKanAdded});
        auto draw = is_kan_draw ? wall_.KanDraw() : wall_.Draw();
        mutable_player(who).Draw(draw);

        last_event_ = Event::CreateDraw(who);
        last_draw_ = last_event_;
        state_.mutable_event_history()->mutable_events()->Add(last_event_.proto());
        state_.mutable_private_infos(ToUType(who))->add_draws(draw.Id());

        return draw;
    }

    void State::Discard(AbsolutePos who, Tile discard) {
        auto [discarded, tsumogiri] = mutable_player(who).Discard(discard);
        assert(discard == discarded);

        last_event_ = Event::CreateDiscard(who, discard, tsumogiri);
        last_discard_ = last_event_;
        state_.mutable_event_history()->mutable_events()->Add(last_event_.proto());
        // TODO: set discarded tile to river
    }

    void State::Riichi(AbsolutePos who) {
        mutable_player(who).Riichi();

        last_event_ = Event::CreateRiichi(who);
        state_.mutable_event_history()->mutable_events()->Add(last_event_.proto());

        require_riichi_score_change_ = true;
    }

    void State::ApplyOpen(AbsolutePos who, Open open) {
        mutable_player(who).ApplyOpen(open);

        last_event_ = Event::CreateOpen(who, open);
        state_.mutable_event_history()->mutable_events()->Add(last_event_.proto());
    }

    void State::AddNewDora() {
        auto [new_dora_ind, new_ura_dora_ind] = wall_.AddKanDora();

        last_event_ = Event::CreateNewDora(new_dora_ind);
        state_.mutable_event_history()->mutable_events()->Add(last_event_.proto());
        state_.add_doras(new_dora_ind.Id());
        state_.add_ura_doras(new_ura_dora_ind.Id());
    }

    void State::RiichiScoreChange() {
        auto who = last_event_.who();
        curr_score_.set_riichi(riichi() + 1);
        curr_score_.set_ten(ToUType(who), ten(who) - 1000);

        last_event_ = Event::CreateRiichiScoreChange(who);
        state_.mutable_event_history()->mutable_events()->Add(last_event_.proto());

        require_riichi_score_change_ = false;
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
        curr_score_.set_riichi(0);

        // set event
        assert(hand_info.win_tile);
        last_event_ = Event::CreateTsumo(winner, hand_info.win_tile.value());
        state_.mutable_event_history()->mutable_events()->Add(last_event_.proto());

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
            curr_score_.set_ten(ToUType(who), ten(who) + ten_move);
        }
        state_.mutable_terminal()->mutable_wins()->Add(std::move(win));
        state_.mutable_terminal()->set_is_game_over(IsGameOver());
        state_.mutable_terminal()->mutable_final_score()->CopyFrom(curr_score_);
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
        curr_score_.set_riichi(0);

        // set event
        last_event_ = Event::CreateRon(winner, tile);
        state_.mutable_event_history()->mutable_events()->Add(last_event_.proto());

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
            curr_score_.set_ten(ToUType(who), ten(who) + ten_move);
        }
        // set win to terminal
        state_.mutable_terminal()->mutable_wins()->Add(std::move(win));
        state_.mutable_terminal()->set_is_game_over(IsGameOver());
        state_.mutable_terminal()->mutable_final_score()->CopyFrom(curr_score_);
    }

    void State::NoWinner() {
        // set event
        last_event_ = Event::CreateNoWinner();
        state_.mutable_event_history()->mutable_events()->Add(last_event_.proto());

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
            curr_score_.set_ten(i, ten(AbsolutePos(i)) + ten_move);
        }
        state_.mutable_terminal()->set_is_game_over(IsGameOver());
        state_.mutable_terminal()->mutable_final_score()->CopyFrom(curr_score_);
    }

    bool State::IsGameOver() const {
        // TODO (sotetsuk): 西入後の終曲条件が供託未収と書いてあるので、修正が必要。　https://tenhou.net/man/
        // ラス親のあがりやめも考慮しないといけない
        auto tens_ = tens();
        bool is_game_over = *std::min_element(tens_.begin(), tens_.end()) < 0 ||
                (round() >= 7 && *std::max_element(tens_.begin(), tens_.end()) >= 30000);
        return is_game_over;
    }

    std::pair<HandInfo, WinScore> State::EvalWinHand(AbsolutePos who) const noexcept {
       return player(who).EvalWinHand(win_state_info(who));
    }

    AbsolutePos State::dealer() const {
        return AbsolutePos(state_.init_score().round() % 4);
    }

    std::uint8_t State::round() const {
        return curr_score_.round();
    }

    std::uint8_t State::honba() const {
        return curr_score_.honba();
    }

    std::uint8_t State::riichi() const {
        return curr_score_.riichi();
    }

    std::array<std::int32_t, 4> State::tens() const {
        std::array<std::int32_t, 4> tens_{};
        for (int i = 0; i < 4; ++i) tens_[i] = curr_score_.ten(i);
        return tens_;
    }

    Wind State::prevalent_wind() const {
        return Wind(round() / 4);
    }

    std::int32_t State::ten(AbsolutePos who) const {
        return curr_score_.ten(ToUType(who));
    }

    State State::Next() const {
        // assert(IsRoundOver());
        assert(!IsGameOver());
        std::vector<PlayerId> player_ids(state_.player_ids().begin(), state_.player_ids().end());
        if (last_event_.type() == EventType::kNoWinner) {
            if (player(dealer()).IsTenpai()) {
                return State(player_ids, seed_, round(), honba() + 1, riichi(), tens());
            } else {
                return State(player_ids, seed_, round() + 1, honba() + 1, riichi(), tens());
            }
        } else {
            if (last_event_.who() == dealer()) {
                return State(player_ids, seed_, round(), honba() + 1, riichi(), tens());
            } else {
                return State(player_ids, seed_, round() + 1, 0, riichi(), tens());
            }
        }
    }

    std::uint8_t State::init_riichi() const {
        return state_.init_score().riichi();
    }

    std::array<std::int32_t, 4> State::init_tens() const {
        std::array<std::int32_t, 4> tens_{};
        for (int i = 0; i < 4; ++i) tens_[i] = state_.init_score().ten(i);
        return tens_;
    }

    std::unordered_map<PlayerId, Observation> State::CreateStealAndRonObservation() const {
        std::unordered_map<PlayerId, Observation> observations;
        auto discarder = last_event_.who();
        auto discard = last_discard_.tile();
        auto has_draw_left = wall_.HasDrawLeft();
        for (int i = 0; i < 4; ++i) {
             auto stealer = AbsolutePos(i);
             if (stealer == discarder) continue;
             auto observation = Observation(stealer, state_);

             // check ron
            if (player(stealer).IsCompleted(discard) &&
                 player(stealer).CanRon(discard, win_state_info(stealer))) {
                 observation.add_possible_action(PossibleAction::CreateRon());
             }

             // check chi, pon and kan_opened
             if (has_draw_left) {
                auto relative_pos = ToRelativePos(stealer, discarder);
                auto possible_opens = player(stealer).PossibleOpensAfterOthersDiscard(discard, relative_pos);
                for (const auto & possible_open: possible_opens)
                    observation.add_possible_action(PossibleAction::CreateOpen(possible_open));
             }

             if (!observation.has_possible_action()) continue;
             observation.add_possible_action(PossibleAction::CreateNo());

             observations[player(stealer).player_id()] = std::move(observation);
         }
         return observations;
    }

    WinStateInfo State::win_state_info(AbsolutePos who) const {
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
        return win_state_info;
    }

    void State::Update(std::vector<Action> &&action_candidates) {
        assert(!action_candidates.empty() && action_candidates.size() <= 3);
        if (action_candidates.size() == 1) {
            Update(std::move(action_candidates.front()));
        } else {
            // if more than 1 action candidates exist
            bool ron_exist = std::count_if(action_candidates.begin(), action_candidates.end(),
                    [](const Action &x){ return x.type() == ActionType::kRon; });
            if (ron_exist) {
                // if ron exist, apply update to all ron action
                // TODO: sort ron actions from 上家 to 下家
                for (auto & action_candidate : action_candidates) Update(std::move(action_candidate));
            } else {
                // if ron does not exist, update via Pon or Kan action
                auto it = find_if(action_candidates.begin(), action_candidates.end(),
                        [](const Action &x){ return x.type() == ActionType::kPon || x.type() == ActionType::kKanOpened; });
                Update(std::move(*it));
            }
        }
    }

    void State::Update(Action &&action) {
        auto who = action.who();
        switch (action.type()) {
            case ActionType::kDiscard:
                {
                    Discard(who, action.discard());
                    // TODO: CreateStealAndRonObservationが2回stateが変わらないのに呼ばれている（CreateObservation内で）
                    bool has_steal_or_ron = !CreateStealAndRonObservation().empty();
                    if (!has_steal_or_ron) {
                        if (wall_.HasDrawLeft()) {
                            if (require_riichi_score_change_) RiichiScoreChange();
                            Draw(AbsolutePos((ToUType(who) + 1) % 4));
                        } else {
                            NoWinner();
                        }
                    }
                }
                break;
            case ActionType::kRiichi:
                Riichi(who);
                break;
            case ActionType::kTsumo:
                Tsumo(who);
                break;
            case ActionType::kRon:
                Ron(who, last_discard_.who(), last_discard_.tile());
                break;
            case ActionType::kChi:
            case ActionType::kPon:
            case ActionType::kKanOpened:
                if (require_riichi_score_change_) RiichiScoreChange();
                ApplyOpen(who, action.open());
                break;
            case ActionType::kKanClosed:
            case ActionType::kKanAdded:
                ApplyOpen(who, action.open());
                break;
            case ActionType::kNo:
                if (wall_.HasDrawLeft()) {
                    if (require_riichi_score_change_) RiichiScoreChange();
                    Draw(AbsolutePos((ToUType(last_discard_.who()) + 1) % 4));  // TODO: check 流局
                } else {
                    NoWinner();
                }
                break;
            case ActionType::kKyushu:
                assert(false);  // Not implemented yet
        }
        last_action_ = std::move(action);
   }
}  // namespace mj
