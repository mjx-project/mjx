#include "state.h"
#include "utils.h"

#include <google/protobuf/util/json_util.h>
#include <google/protobuf/util/message_differencer.h>

namespace mj
{
    State::State(std::vector<PlayerId> player_ids, std::uint32_t seed, int round, int honba, int riichi, std::array<int, 4> tens)
    : seed_(seed), wall_(0, seed) {
        // TODO: use seed_
        assert(std::set<PlayerId>(player_ids.begin(), player_ids.end()).size() == 4);  // player_ids should be identical
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
        return is_round_over_;
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
                {
                    auto who = last_event_.who();
                    auto player_id = player(who).player_id();
                    auto observation = Observation(who, state_);

                    // => NineTiles
                    if (is_first_turn_wo_open && player(who).CanNineTiles()) {
                        observation.add_possible_action(PossibleAction::CreateNineTiles());
                    }

                    // => Tsumo (1)
                    if (player(who).IsCompleted() && player(who).CanTsumo(win_state_info(who)))
                        observation.add_possible_action(PossibleAction::CreateTsumo());

                    // => Kan (2)
                    if (auto possible_kans = player(who).PossibleOpensAfterDraw(); !possible_kans.empty()) {
                        for (const auto possible_kan: possible_kans) {
                            observation.add_possible_action(PossibleAction::CreateOpen(possible_kan));
                        }
                    }

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
                    auto who = last_event_.who();
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
                // => Ron (7)
                // => Chi, Pon and KanOpened (8)
                {
                    assert(!CreateStealAndRonObservation().empty());
                    return CreateStealAndRonObservation();
                }
            case EventType::kKanAdded:
                {
                    auto observations = CreateStealAndRonObservation();
                    assert(!observations.empty());
                    for (const auto &[player_id, observation]: observations)
                        for (const auto &possible_action: observation.possible_actions())
                            assert(Any(possible_action.type(), {ActionType::kRon, ActionType::kNo}));
                    return observations;
                }
            case EventType::kTsumo:
            case EventType::kRon:
            case EventType::kKanClosed:
            case EventType::kKanOpened:
            case EventType::kNoWinner:
            case EventType::kNewDora:
            case EventType::kRiichiScoreChange:
                assert(false);  // Impossible state
        }
    }

    State::State(const std::string &json_str) {
         mjproto::State state = mjproto::State();
         auto status = google::protobuf::util::JsonStringToMessage(json_str, &state);
         assert(status.ok());

         // Set player ids
         state_.mutable_player_ids()->CopyFrom(state.player_ids());
         // Set scores
         state_.mutable_init_score()->CopyFrom(state.init_score());
         curr_score_.CopyFrom(state.init_score());
         // Set walls
         auto wall_tiles = std::vector<Tile>();
         for (auto tile_id: state.wall()) wall_tiles.emplace_back(Tile(tile_id));
         wall_ = Wall(round(), wall_tiles);
         state_.mutable_wall()->CopyFrom(state.wall());
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
         // 三家和了はEventからは復元できないので, ここでSetする
         if (state.terminal().has_no_winner() and state.terminal().no_winner().type() == mjproto::NO_WINNER_TYPE_THREE_RONS) {
             std::vector<int> tenpai = {0, 0, 0, 0};
             for (auto t : state.terminal().no_winner().tenpais()) {
                 tenpai[ToUType(t.who())] = 1;
             }
             assert(std::accumulate(tenpai.begin(), tenpai.end(), 0) == 3);
             for (int i = 0; i < 4; ++i) {
                 if (tenpai[i] == 0) three_ronned_player = AbsolutePos(i);
             }
         }
         // Set event history
         std::vector<int> draw_ixs = {0, 0, 0, 0};
         for (int i = 0; i < state.event_history().events_size(); ++i) {
             auto event = state.event_history().events(i);
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
                     assert(last_event_.type() == EventType::kKanAdded || last_event_.tile() == Tile(event.tile()));
                     Ron(who);
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
         // if (!google::protobuf::util::MessageDifferencer::Equals(state, state_)) {
         //     std::string expected;
         //     google::protobuf::util::MessageToJsonString(state, &expected);
         //     std::cerr << "Expected: " << expected << std::endl;
         //     std::cerr << "Actual  : " << ToJson() << std::endl;
         // }
         // assert(google::protobuf::util::MessageDifferencer::Equals(state, state_));
    }

    std::string State::ToJson() const {
        std::string serialized;
        auto status = google::protobuf::util::MessageToJsonString(state_, &serialized);
        assert(status.ok());
        return serialized;
    }

    Tile State::Draw(AbsolutePos who) {
        if (last_ronable_tile.has_value()) {
            for (int i = 0; i < 4; ++i) {
                auto type = last_ronable_tile.value().Type();
                auto ix = ToUType(type);
                missed_tiles[AbsolutePos(i)].set(ix);
            }
        }
        if (!player(who).IsUnderRiichi()) missed_tiles[who].reset();  // フリテン解除

        auto draw = require_kan_draw_ ? wall_.KanDraw() : wall_.Draw();
        require_kan_draw_ = false;
        mutable_player(who).Draw(draw);

        // 加槓=>槍槓=>Noのときの一発消し。加槓時に自分の一発は外れている外れているはずなので、一発が残っているのは他家のだれか
        if (last_event_.type() == EventType::kKanAdded) for (int i = 0; i < 4; ++i) is_ippatsu_[AbsolutePos(i)] = false;

        last_event_ = Event::CreateDraw(who);
        state_.mutable_event_history()->mutable_events()->Add(last_event_.proto());
        state_.mutable_private_infos(ToUType(who))->add_draws(draw.Id());

        return draw;
    }

    void State::Discard(AbsolutePos who, Tile discard) {
        auto [discarded, tsumogiri] = mutable_player(who).Discard(discard);
        assert(discard == discarded);

        last_ronable_tile = discard; // ロンされうる牌を更新

        is_ippatsu_[who] = false;
        // set is_four_winds = false
        if (is_first_turn_wo_open && is_four_winds) {
            if (!Is(discard.Type(), TileSetType::kWinds)) is_four_winds = false;
            if (dealer() != who && last_discard_type_ != discard.Type()) is_four_winds = false;
        }
        if (Is(discard.Type(), TileSetType::kTanyao)) {
            has_nm[ToUType(who)] = false;
        }

        last_event_ = Event::CreateDiscard(who, discard, tsumogiri);
        last_discard_type_ = discard.Type();
        state_.mutable_event_history()->mutable_events()->Add(last_event_.proto());
        // TODO: set discarded tile to river

        bool is_first_discard_of_north_player = is_first_turn_wo_open && ToSeatWind(who, dealer()) == Wind::kNorth;
        if (is_first_discard_of_north_player) {
            if(is_four_winds) return;  //  go to NoWinner end
            else is_first_turn_wo_open = false;
        }
    }

    void State::Riichi(AbsolutePos who) {
        mutable_player(who).Riichi(is_first_turn_wo_open);

        last_event_ = Event::CreateRiichi(who);
        state_.mutable_event_history()->mutable_events()->Add(last_event_.proto());

        require_riichi_score_change_ = true;
    }

    void State::ApplyOpen(AbsolutePos who, Open open) {
        mutable_player(who).ApplyOpen(open);

        int absolute_pos_from = (ToUType(who) + ToUType(open.From())) % 4;
        has_nm[absolute_pos_from] = false; // 鳴かれた人は流し満貫が成立しない

        if (open.Type() == OpenType::kKanAdded) {
            last_ronable_tile = open.LastTile();    // KanAddedはロンされうる
        }

        last_event_ = Event::CreateOpen(who, open);
        state_.mutable_event_history()->mutable_events()->Add(last_event_.proto());
        if (Any(open.Type(), {OpenType::kKanClosed, OpenType::kKanOpened, OpenType::kKanAdded})) {
            require_kan_draw_ = true;
            ++require_kan_dora_;
        }

        is_first_turn_wo_open = false;
        // 一発解消は「純正巡消しは発声＆和了打診後（加槓のみ)、嶺上ツモの前（連続する加槓の２回目には一発は付かない）」なので、
        // 加槓時は自分の一発だけ消して（一発・嶺上開花は併発しない）、その他のときには全員の一発を消す
        if (open.Type() == OpenType::kKanAdded) is_ippatsu_[who] = false;
        else for (int i = 0; i < 4; ++i) is_ippatsu_[AbsolutePos(i)] = false;
    }

    void State::AddNewDora() {
        auto [new_dora_ind, new_ura_dora_ind] = wall_.AddKanDora();

        last_event_ = Event::CreateNewDora(new_dora_ind);
        state_.mutable_event_history()->mutable_events()->Add(last_event_.proto());
        state_.add_doras(new_dora_ind.Id());
        state_.add_ura_doras(new_ura_dora_ind.Id());

        --require_kan_dora_;
    }

    void State::RiichiScoreChange() {
        auto who = last_event_.who();
        curr_score_.set_riichi(riichi() + 1);
        curr_score_.set_ten(ToUType(who), ten(who) - 1000);

        last_event_ = Event::CreateRiichiScoreChange(who);
        state_.mutable_event_history()->mutable_events()->Add(last_event_.proto());

        require_riichi_score_change_ = false;
        is_ippatsu_[who] = true;
    }

    void State::Tsumo(AbsolutePos winner) {
        mutable_player(winner).Tsumo();
        auto [hand_info, win_score] = EvalWinHand(winner);
        // calc ten moves
        auto pao = (win_score.HasYakuman(Yaku::kBigThreeDragons) || win_score.HasYakuman(Yaku::kBigFourWinds)) ? HasPao(winner) : std::nullopt;
        auto ten_moves = win_score.TenMoves(winner, dealer());
        auto ten_ = ten_moves[winner];
        if (pao) {  // 大三元・大四喜の責任払い
            assert(pao.value() != winner);
            for (auto &[who, ten_move]: ten_moves) {
                if (ten_move > 0) ten_move += riichi() * 1000 + honba() * 300;
                else if (pao.value() == who) ten_move = - ten_ - honba() * 300;
                else ten_move = 0;
            }
        } else {
            for (auto &[who, ten_move]: ten_moves) {
                if (ten_move > 0) ten_move += riichi() * 1000 + honba() * 300;
                else if (ten_move < 0) ten_move -= honba() * 100;
            }
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
        std::reverse(hand_info.opens.begin(), hand_info.opens.end());  // To follow tenhou's format
        for (const auto &open: hand_info.opens) {
            win.add_opens(open.GetBits());
        }
        assert(hand_info.win_tile);
        win.set_win_tile(hand_info.win_tile.value().Id());
        // fu
        if (win_score.fu()) win.set_fu(win_score.fu().value());
        else win.set_fu(0);  // 役満のとき形式上0としてセットする
        // yaku, fans
        std::vector<std::pair<Yaku, std::uint8_t>> yakus;
        for (const auto &[yaku, fan]: win_score.yaku()) {
            win.add_yakus(ToUType(yaku));
            win.add_fans(fan);
        }
        // yakumans
        for (const auto& yakuman: win_score.yakuman()) {
            win.add_yakumans(ToUType(yakuman));
        }
        // ten and ten moves
        win.set_ten(ten_);
        for (int i = 0; i < 4; ++i) win.add_ten_changes(0);
        for (const auto &[who, ten_move]: ten_moves) {
            win.set_ten_changes(ToUType(who), ten_move);
            curr_score_.set_ten(ToUType(who), ten(who) + ten_move);
        }

        // set terminal
        is_round_over_ = true;
        if (IsGameOver()) {
            AbsolutePos top = top_player();
            curr_score_.set_ten(ToUType(top), curr_score_.ten(ToUType(top)) + 1000 * riichi());
            curr_score_.set_riichi(0);
        }
        state_.mutable_terminal()->mutable_wins()->Add(std::move(win));
        state_.mutable_terminal()->set_is_game_over(IsGameOver());
        state_.mutable_terminal()->mutable_final_score()->CopyFrom(curr_score_);
    }

    void State::Ron(AbsolutePos winner) {
        assert(Any(last_event_.type(), {EventType::kDiscardDrawnTile, EventType::kDiscardFromHand, EventType::kRiichiScoreChange, EventType::kKanAdded, EventType::kRon}));
        AbsolutePos loser = last_event_.type() != EventType::kRon ? last_event_.who() : AbsolutePos(state_.terminal().wins(0).from_who());
        Tile tile = last_event_.type() != EventType::kKanAdded ? last_event_.tile() : last_event_.open().LastTile();

        mutable_player(winner).Ron(tile);
        auto [hand_info, win_score] = EvalWinHand(winner);
        // calc ten moves
        auto pao = (win_score.HasYakuman(Yaku::kBigThreeDragons) || win_score.HasYakuman(Yaku::kBigFourWinds)) ? HasPao(winner) : std::nullopt;
        auto ten_moves = win_score.TenMoves(winner, dealer(), loser);
        auto ten_ = ten_moves[winner];
        if (pao) {  // 大三元・大四喜の責任払い
            assert(pao.value() != winner);
            for (auto &[who, ten_move]: ten_moves) {
                // TODO: パオかつダブロン時の積み棒も上家取りでいいのか？
                int honba_ = last_event_.type() == EventType::kRon ? 0 : honba();
                int riichi_ = last_event_.type() == EventType::kRon ? 0 : riichi();
                if (ten_move > 0) ten_move += riichi_ * 1000 + honba_ * 300;
                else if (ten_move < 0) ten_move = - (ten_ / 2);
                if (who == pao.value()) ten_move -= ((ten_ / 2) + honba_ * 300);  // 積み棒はパオが払う。パオがロンされたときに注意
            }
        } else {
            for (auto &[who, ten_move]: ten_moves) {
                // ダブロンは上家取り
                int honba_ = last_event_.type() == EventType::kRon ? 0 : honba();
                int riichi_ = last_event_.type() == EventType::kRon ? 0 : riichi();
                if (ten_move > 0) ten_move += riichi_ * 1000 + honba_ * 300;
                else if (ten_move < 0) ten_move -= honba_ * 300;
            }
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
        std::reverse(hand_info.opens.begin(), hand_info.opens.end());  // To follow tenhou's format
        for (const auto &open: hand_info.opens) {
            win.add_opens(open.GetBits());
        }
        win.set_win_tile(tile.Id());
        // fu
        if (win_score.fu()) win.set_fu(win_score.fu().value());
        else win.set_fu(0);  // 役満のとき形式上0としてセットする
        // yaku, fans
        std::vector<std::pair<Yaku, std::uint8_t>> yakus;
        for (const auto &[yaku, fan]: win_score.yaku()) {
            win.add_yakus(ToUType(yaku));
            win.add_fans(fan);
        }
        // yakumans
        for (const auto& yakuman: win_score.yakuman()) {
            win.add_yakumans(ToUType(yakuman));
        }
        // ten and ten moves
        win.set_ten(ten_);
        for (int i = 0; i < 4; ++i) win.add_ten_changes(0);
        for (const auto &[who, ten_move]: ten_moves) {
            win.set_ten_changes(ToUType(who), ten_move);
            curr_score_.set_ten(ToUType(who), ten(who) + ten_move);
        }

        // set win to terminal
        is_round_over_ = true;
        if (IsGameOver()) {
            AbsolutePos top = top_player();
            curr_score_.set_ten(ToUType(top), curr_score_.ten(ToUType(top)) + 1000 * riichi());
            curr_score_.set_riichi(0);
        }
        state_.mutable_terminal()->mutable_wins()->Add(std::move(win));
        state_.mutable_terminal()->set_is_game_over(IsGameOver());
        state_.mutable_terminal()->mutable_final_score()->CopyFrom(curr_score_);
    }

    void State::NoWinner() {
        // 四家立直, 三家和了, 四槓散了, 流し満貫
        auto set_terminal_vals = [&]() {
            state_.mutable_terminal()->mutable_final_score()->CopyFrom(curr_score_);
            for (int i = 0; i < 4; ++i) state_.mutable_terminal()->mutable_no_winner()->add_ten_changes(0);
            last_event_ = Event::CreateNoWinner();
            state_.mutable_event_history()->mutable_events()->Add(last_event_.proto());
            is_round_over_ = true;
        };
        // 九種九牌
        if (is_first_turn_wo_open && last_event_.type() == EventType::kDraw) {
            state_.mutable_terminal()->mutable_no_winner()->set_type(mjproto::NO_WINNER_TYPE_KYUUSYU);
            mjproto::TenpaiHand tenpai;
            tenpai.set_who(mjproto::AbsolutePos(last_event_.who()));
            for (auto tile: player(last_event_.who()).closed_tiles()) tenpai.mutable_closed_tiles()->Add(tile.Id());
            state_.mutable_terminal()->mutable_no_winner()->mutable_tenpais()->Add(std::move(tenpai));
            set_terminal_vals();
            return;
        }
        // 四風子連打
        if (is_first_turn_wo_open && is_four_winds) {
            state_.mutable_terminal()->mutable_no_winner()->set_type(mjproto::NO_WINNER_TYPE_FOUR_WINDS);
            set_terminal_vals();
            return;
        }
        // 四槓散了
        if (IsFourKanNoWinner()) {
            state_.mutable_terminal()->mutable_no_winner()->set_type(mjproto::NO_WINNER_TYPE_FOUR_KANS);
            set_terminal_vals();
            return;
        }
        // 三家和了
        if (three_ronned_player) {
            state_.mutable_terminal()->mutable_no_winner()->set_type(mjproto::NO_WINNER_TYPE_THREE_RONS);
            // 聴牌の情報が必要なため, ここでreturnしてはいけない.
        }

        // 四家立直
        if (std::all_of(players_.begin(), players_.end(), [](const Player& p){ return p.IsUnderRiichi(); })) {
            state_.mutable_terminal()->mutable_no_winner()->set_type(mjproto::NO_WINNER_TYPE_FOUR_RIICHI);
            // 聴牌の情報が必要なため, ここでreturnしてはいけない.
        }

        // set event
        last_event_ = Event::CreateNoWinner();
        state_.mutable_event_history()->mutable_events()->Add(last_event_.proto());

        // set terminal
        std::vector<int> is_tenpai = {0, 0, 0, 0};
        for (int i = 0; i < 4; ++i) {
            auto who = AbsolutePos(i);
            if (three_ronned_player and three_ronned_player.value() == who) continue; // 三家和了でロンされた人の聴牌情報は入れない
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

        std::vector<int> ten_move{0, 0, 0, 0};
        // 流し満貫
        if (std::any_of(has_nm.begin(), has_nm.end(), [](bool hasnm){ return hasnm; })) {
            int dealer_ix = ToUType(dealer());
            for (int i = 0; i < 4; ++i) {
                if (has_nm[i]) {
                    for (int j = 0; j < 4; ++j) {
                        if (i == j) ten_move[j] += (i == dealer_ix ? 12000 : 8000);
                        else        ten_move[j] -= (i == dealer_ix or j == dealer_ix ? 4000 : 2000);
                    }
                }
            }
            state_.mutable_terminal()->mutable_no_winner()->set_type(mjproto::NO_WINNER_TYPE_NM);
        }
        else if (!three_ronned_player) {
            auto num_tenpai = std::accumulate(is_tenpai.begin(), is_tenpai.end(), 0);
            for (int i = 0; i < 4; ++i) {
                switch (num_tenpai) {
                    case 1:
                        ten_move[i] = is_tenpai[i] ? 3000 : -1000;
                        break;
                    case 2:
                        ten_move[i] = is_tenpai[i] ? 1500 : -1500;
                        break;
                    case 3:
                        ten_move[i] = is_tenpai[i] ? 1000 : -3000;
                        break;
                    default:  // 0, 4
                        ten_move[i] = 0;
                        break;
                }
            }
        }

        // apply ten moves
        for (int i = 0; i < 4; ++i) {
            state_.mutable_terminal()->mutable_no_winner()->add_ten_changes(ten_move[i]);
            curr_score_.set_ten(i, ten(AbsolutePos(i)) + ten_move[i]);
        }

        // set terminal
        is_round_over_ = true;
        if (IsGameOver()) {
            AbsolutePos top = top_player();
            curr_score_.set_ten(ToUType(top), curr_score_.ten(ToUType(top)) + 1000 * riichi());
            curr_score_.set_riichi(0);
        }
        state_.mutable_terminal()->set_is_game_over(IsGameOver());
        state_.mutable_terminal()->mutable_final_score()->CopyFrom(curr_score_);
    }

    bool State::IsGameOver() const {
        if (!IsRoundOver()) return false;

        auto tens_ = tens();
        for (int i = 0; i < 4; ++i) tens_[i] += 4 - i;  // 同点は起家から順に優先されるので +4, +3, +2, +1 する
        auto top_score = *std::max_element(tens_.begin(), tens_.end());

        // 箱割れ
        bool has_minus_point_player = *std::min_element(tens_.begin(), tens_.end()) < 0;
        if (has_minus_point_player) return true;

        // 東南戦
        if (round() < 7) return false;

        // 北入なし
        bool dealer_win_or_tenpai = (Any(last_event_.type(), {EventType::kRon, EventType::kTsumo})
                                     && std::any_of(state_.terminal().wins().begin(), state_.terminal().wins().end(), [&](const auto x){ return AbsolutePos(x.who()) == dealer(); })) ||
                                    (last_event_.type() == EventType::kNoWinner && player(dealer()).IsTenpai());
        if (round() == 11 && !dealer_win_or_tenpai) return true;

        // トップが3万点必要（供託未収）
        bool top_has_30000 = *std::max_element(tens_.begin(), tens_.end()) >= 30000;
        if (!top_has_30000) return false;

        // オーラストップ親の上がりやめあり
        bool dealer_is_not_top = top_score != tens_[ToUType(dealer())];
        return !(dealer_win_or_tenpai && dealer_is_not_top);
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
        auto tile = last_event_.type() != EventType::kKanAdded ? last_event_.tile() : last_event_.open().LastTile();
        auto has_draw_left = wall_.HasDrawLeft();

        for (int i = 0; i < 4; ++i) {
            auto stealer = AbsolutePos(i);
            if (stealer == discarder) continue;
             auto observation = Observation(stealer, state_);

             // check ron
             if (player(stealer).IsCompleted(tile) &&
                 player(stealer).CanRon(tile, win_state_info(stealer), missed_tiles.at(stealer))) {
                 observation.add_possible_action(PossibleAction::CreateRon());
             }

             // check chi, pon and kan_opened
             if (has_draw_left && last_event_.type() != EventType::kKanAdded && !IsFourKanNoWinner()) {  // if 槍槓 or 四槓散了直前の捨て牌, only ron
                auto relative_pos = ToRelativePos(stealer, discarder);
                auto possible_opens = player(stealer).PossibleOpensAfterOthersDiscard(tile, relative_pos);
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
                !wall_.HasDrawLeft(),
                is_ippatsu_.at(who),
                is_first_turn_wo_open && last_event_.who() == who
                        && (Any(last_event_.type(), {EventType::kDraw, EventType::kTsumo})),
                seat_wind == Wind::kEast,
                last_event_.type() == EventType::kKanAdded, // robbing kan
                wall_.dora_count(),
                wall_.ura_dora_count());
        return win_state_info;
    }

    void State::Update(std::vector<Action> &&action_candidates) {
        static_assert(ActionType::kNo < ActionType::kChi);
        static_assert(ActionType::kChi < ActionType::kPon);
        static_assert(ActionType::kChi < ActionType::kKanOpened);
        static_assert(ActionType::kPon < ActionType::kRon);
        static_assert(ActionType::kKanOpened < ActionType::kRon);
        assert(!action_candidates.empty() && action_candidates.size() <= 3);

        if (action_candidates.size() == 1) {
            Update(std::move(action_candidates.front()));
        } else {
            // sort in order Ron > KanOpened > Pon > Chi > No
            std::sort(action_candidates.begin(), action_candidates.end(),
                    [](const Action &x, const Action &y){ return x.type() > x.type(); });
            bool has_ron = action_candidates.front().type() == ActionType::kRon;
            if (has_ron) {
                int ron_count = std::count_if(action_candidates.begin(), action_candidates.end(),
                                               [](const Action &x){ return x.type() == ActionType::kRon; });
                if (ron_count == 3) {
                    // 三家和了
                    std::vector<int> ron = {0, 0, 0, 0};
                    for (auto action : action_candidates) {
                        if (action.type() == ActionType::kRon) ron[ToUType(action.who())] = 1;
                    }
                    assert(std::accumulate(ron.begin(), ron.end(), 0) == 3);
                    for (int i = 0; i < 4; ++i) {
                        if (ron[i] == 0) three_ronned_player = AbsolutePos(i);
                    }
                    NoWinner();
                    return;
                }
                for (auto &action: action_candidates) {
                    if (action.type() != ActionType::kRon) break;
                    Update(std::move(action));
                }
            } else {
                assert(Any(action_candidates.front().type(), {ActionType::kNo, ActionType::kChi, ActionType::kPon, ActionType::kKanOpened}));
                Update(std::move(action_candidates.front()));
            }
        }
    }

    void State::Update(Action &&action) {
        assert(Any(last_event_.type(), {EventType::kDraw, EventType::kDiscardFromHand,EventType::kDiscardDrawnTile,
                                        EventType::kRiichi, EventType::kChi, EventType::kPon, EventType::kKanAdded}));
        auto who = action.who();
        switch (action.type()) {
            case ActionType::kDiscard:
                {
                    assert(Any(last_event_.type(), {EventType::kDraw, EventType::kChi, EventType::kRon, EventType::kRiichi}));
                    assert(require_kan_dora_ <= 1);
                    if (require_kan_dora_) AddNewDora();
                    Discard(who, action.discard());
                    if (is_first_turn_wo_open && ToSeatWind(who, dealer()) == Wind::kNorth && is_four_winds) {  // 四風子連打
                        NoWinner();
                        return;
                    }
                    // TODO: CreateStealAndRonObservationが2回stateが変わらないのに呼ばれている（CreateObservation内で）
                    if (bool has_steal_or_ron = !CreateStealAndRonObservation().empty(); has_steal_or_ron) return;

                    // 鳴きやロンの候補がなく, 全員が立直していたら四家立直で流局
                    if (std::all_of(players_.begin(), players_.end(),
                                    [](const Player& player){ return player.IsUnderRiichi(); })) {
                        RiichiScoreChange();
                        NoWinner();
                        return;
                    }
                    // 鳴きやロンの候補がなく, 2人以上が合計4つ槓をしていたら四槓散了で流局
                    {
                        std::vector<int> kans;
                        for (const Player& p : players_) {
                            if (int num = p.TotalKans(); num) kans.emplace_back(num);
                        }
                        if (std::accumulate(kans.begin(), kans.end(), 0) == 4 and kans.size() > 1) {
                            NoWinner();
                            return;
                        }
                    }

                    if (wall_.HasDrawLeft()) {
                        if (require_riichi_score_change_) RiichiScoreChange();
                        Draw(AbsolutePos((ToUType(who) + 1) % 4));
                    } else {
                        NoWinner();
                    }
                }
                return;
            case ActionType::kRiichi:
                assert(Any(last_event_.type(), {EventType::kDraw}));
                Riichi(who);
                return;
            case ActionType::kTsumo:
                assert(Any(last_event_.type(), {EventType::kDraw}));
                Tsumo(who);
                return;
            case ActionType::kRon:
                assert(Any(last_event_.type(), {EventType::kDiscardFromHand, EventType::kDiscardDrawnTile, EventType::kKanAdded}));
                Ron(who);
                return;
            case ActionType::kChi:
            case ActionType::kPon:
                assert(Any(last_event_.type(), {EventType::kDiscardFromHand, EventType::kDiscardDrawnTile}));
                if (require_riichi_score_change_) RiichiScoreChange();
                ApplyOpen(who, action.open());
                return;
            case ActionType::kKanOpened:
                assert(Any(last_event_.type(), {EventType::kDiscardFromHand, EventType::kDiscardDrawnTile}));
                if (require_riichi_score_change_) RiichiScoreChange();
                ApplyOpen(who, action.open());
                Draw(who);
                return;
            case ActionType::kKanClosed:
                assert(Any(last_event_.type(), {EventType::kDraw}));
                ApplyOpen(who, action.open());
                // 天鳳のカンの仕様については https://github.com/sotetsuk/mahjong/issues/199 で調べている
                // 暗槓の分で最低一回は新ドラがめくられる
                assert(require_kan_dora_ <= 2);
                while(require_kan_dora_) AddNewDora();
                Draw(who);
                return;
            case ActionType::kKanAdded:
                assert(Any(last_event_.type(), {EventType::kDraw}));
                ApplyOpen(who, action.open());
                // TODO: CreateStealAndRonObservationが状態変化がないのに2回計算されている
                if (auto has_no_ron = CreateStealAndRonObservation().empty(); has_no_ron) {
                    assert(require_kan_dora_ <= 2);
                    while(require_kan_dora_ > 1) AddNewDora();  // 前のカンの分の新ドラをめくる。1回分はここでの加槓の分なので、ここではめくられない
                    Draw(who);
                }
                return;
            case ActionType::kNo:
                assert(Any(last_event_.type(), {EventType::kDiscardDrawnTile, EventType::kDiscardFromHand, EventType::kKanAdded}));

                // 加槓のあとに ActionType::kNo が渡されるのは槍槓のロンを否定した場合のみ
                if (last_event_.type() == EventType::kKanAdded) {
                    Draw(AbsolutePos((ToUType(last_event_.who()))));  // 嶺上ツモ
                    return;
                }

                // 全員が立直している状態で ActionType::kNo が渡されるのは,
                // 4人目に立直した人の立直宣言牌を他家がロンできるけど無視したときのみ.
                // 四家立直で流局とする.
                if (std::all_of(players_.begin(), players_.end(),
                                [](const Player& player){ return player.IsUnderRiichi(); })) {
                    RiichiScoreChange();
                    NoWinner();
                    return;
                }

                // 2人以上が合計4つ槓をしている状態で ActionType::kNo が渡されるのは,
                // 4つ目の槓をした人の打牌を他家がロンできるけど無視したときのみ.
                // 四槓散了で流局とする.
                if (IsFourKanNoWinner()) {
                    NoWinner();
                    return;
                }

                if (wall_.HasDrawLeft()) {
                    if (require_riichi_score_change_) RiichiScoreChange();
                    Draw(AbsolutePos((ToUType(last_event_.who()) + 1) % 4));
                } else {
                    NoWinner();
                }
                return;
            case ActionType::kKyushu:
                assert(Any(last_event_.type(), {EventType::kDraw}));
                NoWinner();
                return;
        }
   }

    AbsolutePos State::top_player() const {
        int top_ix = 0; int top_ten  = INT_MIN;
        for (int i = 0; i < 4; ++i) {
            int ten = curr_score_.ten(i) + (4 - i);  // 同着なら起家から順に優先のため +4, +3, +2, +1
            if (top_ten < ten) {
                top_ix = i;
                top_ten = ten;
            }
        }
        return AbsolutePos(top_ix);
    }

    bool State::IsFourKanNoWinner() const noexcept {
        std::vector<int> kans;
        for (const Player& p : players_) {
            if (int num = p.TotalKans(); num) kans.emplace_back(num);
        }
        return std::accumulate(kans.begin(), kans.end(), 0) == 4 and kans.size() > 1;
    }

    mjproto::State State::proto() const {
        return state_;
    }

    std::optional<AbsolutePos> State::HasPao(AbsolutePos winner) const noexcept {
        auto pao = player(winner).HasPao();
        if (pao) return AbsolutePos((ToUType(winner) + ToUType(pao.value())) % 4);
        else return std::nullopt;
    }
}  // namespace mj
