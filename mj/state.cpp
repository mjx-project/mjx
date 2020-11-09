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
        for (int i = 0; i < 4; ++i) {
            auto hand = Hand(wall_.initial_hand_tiles(AbsolutePos(i)));
            players_[i] = Player{player_ids[i], AbsolutePos(i), std::move(hand)};
        }

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
        for (int i = 0; i < 4; ++i) {
            state_.add_private_infos()->set_who(mjproto::AbsolutePos(i));
            for (const auto tile: wall_.initial_hand_tiles(AbsolutePos(i)))
                state_.mutable_private_infos(i)->mutable_init_hand()->Add(tile.Id());
        }

        // dealer draws the first tusmo
        Draw(dealer());
    }

    bool State::IsRoundOver() const {
        return is_round_over_;
    }


    const State::Player &State::player(AbsolutePos pos) const {
        return players_.at(ToUType(pos));
    }

    State::Player& State::mutable_player(AbsolutePos pos) {
        return players_.at(ToUType(pos));
    }

    const Hand &State::hand(AbsolutePos who) const {
        return player(who).hand;
    }

    Hand& State::mutable_hand(AbsolutePos who) {
        return mutable_player(who).hand;
    }

    GameResult State::result() const {
        // 順位
        const auto final_tens = tens();
        std::vector<std::pair<int, int>> pos_ten;
        for (int i = 0; i < 4; ++i) {
            pos_ten.emplace_back(i, final_tens[i] + (4 - i));  // 同点は起家から順に優先されるので +4, +3, +2, +1 する
        }
        std::sort(pos_ten.begin(), pos_ten.end(), [](auto x, auto y){ return x.second < y.second; });
        std::reverse(pos_ten.begin(), pos_ten.end());
        for (int i = 0; i < 3; ++i) assert(pos_ten[i].second > pos_ten[i + 1].second);
        std::map<PlayerId, int> rankings;
        for (int i = 0; i < 4; ++i) {
            int ranking = i + 1;
            PlayerId player_id = player(AbsolutePos(pos_ten[i].first)).player_id;
            rankings[player_id] = ranking;
        }

        // 点数
        std::map<PlayerId, int> tens_map;
        for (int i = 0; i < 4; ++i) {
            PlayerId player_id = player(AbsolutePos(i)).player_id;
            int ten = final_tens[i];
            tens_map[player_id] = ten;
        }

        return GameResult{0, rankings, tens_map};
    }

    std::unordered_map<PlayerId, Observation> State::CreateObservations() const {
        switch (LastEvent().type()) {
            case mjproto::EVENT_TYPE_DRAW:
                {
                    auto who = LastEvent().who();
                    auto player_id = player(who).player_id;
                    auto observation = Observation(who, state_);

                    // => NineTiles
                    if (IsFirstTurnWithoutOpen() && hand(who).CanNineTiles()) {
                        observation.add_possible_action(Action::CreateNineTiles(who));
                    }

                    // => Tsumo (1)
                    if (hand(who).IsCompleted() && CanTsumo(who))
                        observation.add_possible_action(Action::CreateTsumo(who));

                    // => Kan (2)
                    if (auto possible_kans = hand(who).PossibleOpensAfterDraw(); !possible_kans.empty()) {
                        for (const auto possible_kan: possible_kans) {
                            observation.add_possible_action(Action::CreateOpen(who, possible_kan));
                        }
                    }

                    // => Riichi (3)
                    if (CanRiichi(who))
                        observation.add_possible_action(Action::CreateRiichi(who));

                    // => Discard (4)
                    observation.add_possible_actions(Action::CreateDiscards(who, hand(who).PossibleDiscards()));

                    return { {player_id, std::move(observation)} };
                }
            case mjproto::EVENT_TYPE_RIICHI:
                {
                    // => Discard (5)
                    auto who = LastEvent().who();
                    auto observation = Observation(who, state_);
                    observation.add_possible_actions(Action::CreateDiscards(who, hand(who).PossibleDiscardsJustAfterRiichi()));
                    return { {player(who).player_id, std::move(observation)} };
                }
            case mjproto::EVENT_TYPE_CHI:
            case mjproto::EVENT_TYPE_PON:
                {
                    // => Discard (6)
                    auto who = LastEvent().who();
                    auto observation = Observation(who, state_);
                    observation.add_possible_actions(Action::CreateDiscards(who, hand(who).PossibleDiscards()));
                    return { {player(who).player_id, std::move(observation)} };
                }
            case mjproto::EVENT_TYPE_DISCARD_FROM_HAND:
            case mjproto::EVENT_TYPE_DISCARD_DRAWN_TILE:
                // => Ron (7)
                // => Chi, Pon and KanOpened (8)
                {
                    return CreateStealAndRonObservation();
                }
            case mjproto::EVENT_TYPE_KAN_ADDED:
                {
                    auto observations = CreateStealAndRonObservation();
                    assert(!observations.empty());
                    for (const auto &[player_id, observation]: observations)
                        for (const auto &possible_action: observation.possible_actions())
                            assert(Any(possible_action.type(), {mjproto::ACTION_TYPE_RON,
                                                                mjproto::ACTION_TYPE_NO}));
                    return observations;
                }
            case mjproto::EVENT_TYPE_TSUMO:
            case mjproto::EVENT_TYPE_RON:
            case mjproto::EVENT_TYPE_KAN_CLOSED:
            case mjproto::EVENT_TYPE_KAN_OPENED:
            case mjproto::EVENT_TYPE_NO_WINNER:
            case mjproto::EVENT_TYPE_NEW_DORA:
            case mjproto::EVENT_TYPE_RIICHI_SCORE_CHANGE:
                assert(false);  // Impossible state
        }
    }

    mjproto::State State::LoadJson(const std::string &json_str) {
        mjproto::State state = mjproto::State();
        auto status = google::protobuf::util::JsonStringToMessage(json_str, &state);
        assert(status.ok());
        return state;
    }

    State::State(const std::string &json_str) : State(LoadJson(json_str)) {}

    State::State(const mjproto::State& state) {
        //mjproto::State state = mjproto::State();
        //auto status = google::protobuf::util::JsonStringToMessage(json_str, &state);
        //assert(status.ok());

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
            players_[i] = Player{state_.player_ids(i), AbsolutePos(i), Hand(wall_.initial_hand_tiles(AbsolutePos(i)))};
            state_.mutable_private_infos()->Add();
            state_.mutable_private_infos(i)->set_who(mjproto::AbsolutePos(i));
            for (auto t: wall_.initial_hand_tiles(AbsolutePos(i))) {
                state_.mutable_private_infos(i)->add_init_hand(t.Id());
            }
        }
        // 三家和了はEventからは復元できないので, ここでSetする
        if (state.terminal().has_no_winner() and
            state.terminal().no_winner().type() == mjproto::NO_WINNER_TYPE_THREE_RONS) {
            std::vector<int> tenpai = {0, 0, 0, 0};
            for (auto t : state.terminal().no_winner().tenpais()) {
                tenpai[ToUType(t.who())] = 1;
            }
            assert(std::accumulate(tenpai.begin(), tenpai.end(), 0) == 3);
            for (int i = 0; i < 4; ++i) {
                if (tenpai[i] == 0) three_ronned_player = AbsolutePos(i);
            }
        }

        for (const auto &event : state.event_history().events()) {
            UpdateByEvent(event);
        }
    }

    void State::UpdateByEvent(const mjproto::Event& event) {
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
                assert(LastEvent().type() == mjproto::EVENT_TYPE_KAN_ADDED || LastEvent().tile() == Tile(event.tile()));
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

    std::string State::ToJson() const {
        std::string serialized;
        auto status = google::protobuf::util::MessageToJsonString(state_, &serialized);
        assert(status.ok());
        return serialized;
    }

    Tile State::Draw(AbsolutePos who) {
        if (TargetTile().has_value()) {
            for (int i = 0; i < 4; ++i) {
                auto type = TargetTile().value().Type();
                auto ix = ToUType(type);
                mutable_player(AbsolutePos(i)).missed_tiles.set(ix);
            }
        }
        if (!hand(who).IsUnderRiichi()) mutable_player(who).missed_tiles.reset();  // フリテン解除

        auto draw = require_kan_draw_ ? wall_.KanDraw() : wall_.Draw();
        require_kan_draw_ = false;
        mutable_hand(who).Draw(draw);

        // 加槓=>槍槓=>Noのときの一発消し。加槓時に自分の一発は外れている外れているはずなので、一発が残っているのは他家のだれか
        if (HasLastEvent() and LastEvent().type() == mjproto::EVENT_TYPE_KAN_ADDED) for (int i = 0; i < 4; ++i) mutable_player(AbsolutePos(i)).is_ippatsu = false;
        // 槍槓
        is_robbing_kan = false;

        state_.mutable_event_history()->mutable_events()->Add(Event::CreateDraw(who).proto());
        state_.mutable_private_infos(ToUType(who))->add_draws(draw.Id());

        return draw;
    }

    void State::Discard(AbsolutePos who, Tile discard) {
        mutable_player(who).discards.set(ToUType(discard.Type()));
        auto [discarded, tsumogiri] = mutable_hand(who).Discard(discard);
        if (hand(who).IsTenpai()) {
            mutable_player(who).machi.reset();
            for (auto tile_type : WinHandCache::instance().Machi(hand(who).ClosedTileTypes())) {
                mutable_player(who).machi.set(ToUType(tile_type));
            }
        }
        assert(discard == discarded);

        mutable_player(who).is_ippatsu = false;
        if (Is(discard.Type(), TileSetType::kTanyao)) {
            mutable_player(who).has_nm=false;
        }
        state_.mutable_event_history()->mutable_events()->Add(Event::CreateDiscard(who, discard, tsumogiri).proto());
        // TODO: set discarded tile to river
    }

    void State::Riichi(AbsolutePos who) {
        assert(ten(who) >= 1000);
        assert(wall_.HasNextDrawLeft());
        mutable_hand(who).Riichi(IsFirstTurnWithoutOpen());

        state_.mutable_event_history()->mutable_events()->Add(Event::CreateRiichi(who).proto());

        require_riichi_score_change_ = true;
    }

    void State::ApplyOpen(AbsolutePos who, Open open) {
        mutable_player(who).missed_tiles.reset();  // フリテン解除

        mutable_hand(who).ApplyOpen(open);

        int absolute_pos_from = (ToUType(who) + ToUType(open.From())) % 4;
        mutable_player(AbsolutePos(absolute_pos_from)).has_nm = false; // 鳴かれた人は流し満貫が成立しない

        state_.mutable_event_history()->mutable_events()->Add(Event::CreateOpen(who, open).proto());
        if (Any(open.Type(), {OpenType::kKanClosed, OpenType::kKanOpened, OpenType::kKanAdded})) {
            require_kan_draw_ = true;
            ++require_kan_dora_;
        }

        // 一発解消は「純正巡消しは発声＆和了打診後（加槓のみ)、嶺上ツモの前（連続する加槓の２回目には一発は付かない）」なので、
        // 加槓時は自分の一発だけ消して（一発・嶺上開花は併発しない）、その他のときには全員の一発を消す
        if (open.Type() == OpenType::kKanAdded) {
            mutable_player(who).is_ippatsu = false;
            is_robbing_kan = true;  // 槍槓
        } else {
            for (int i = 0; i < 4; ++i) mutable_player(AbsolutePos(i)).is_ippatsu = false;
        }
    }

    void State::AddNewDora() {
        auto [new_dora_ind, new_ura_dora_ind] = wall_.AddKanDora();

        state_.mutable_event_history()->mutable_events()->Add(Event::CreateNewDora(new_dora_ind).proto());
        state_.add_doras(new_dora_ind.Id());
        state_.add_ura_doras(new_ura_dora_ind.Id());

        --require_kan_dora_;
    }

    void State::RiichiScoreChange() {
        auto who = LastEvent().who();
        curr_score_.set_riichi(riichi() + 1);
        curr_score_.set_ten(ToUType(who), ten(who) - 1000);

        state_.mutable_event_history()->mutable_events()->Add(Event::CreateRiichiScoreChange(who).proto());

        require_riichi_score_change_ = false;
        mutable_player(who).is_ippatsu=true;
    }

    void State::Tsumo(AbsolutePos winner) {
        mutable_player(winner).hand.Tsumo();
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
        state_.mutable_event_history()->mutable_events()->Add(Event::CreateTsumo(winner, hand_info.win_tile.value()).proto());

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
        assert(Any(LastEvent().type(), {
            mjproto::EVENT_TYPE_DISCARD_DRAWN_TILE, mjproto::EVENT_TYPE_DISCARD_FROM_HAND,
            mjproto::EVENT_TYPE_KAN_ADDED, mjproto::EVENT_TYPE_RON}));
        AbsolutePos loser = LastEvent().type() != mjproto::EVENT_TYPE_RON ? LastEvent().who() : AbsolutePos(state_.terminal().wins(0).from_who());
        Tile tile = LastEvent().type() != mjproto::EVENT_TYPE_KAN_ADDED ? LastEvent().tile() : LastEvent().open().LastTile();

        mutable_player(winner).hand.Ron(tile);
        auto [hand_info, win_score] = EvalWinHand(winner);
        // calc ten moves
        auto pao = (win_score.HasYakuman(Yaku::kBigThreeDragons) || win_score.HasYakuman(Yaku::kBigFourWinds)) ? HasPao(winner) : std::nullopt;
        auto ten_moves = win_score.TenMoves(winner, dealer(), loser);
        auto ten_ = ten_moves[winner];
        if (pao) {  // 大三元・大四喜の責任払い
            assert(pao.value() != winner);
            for (auto &[who, ten_move]: ten_moves) {
                // TODO: パオかつダブロン時の積み棒も上家取りでいいのか？
                int honba_ = LastEvent().type() == mjproto::EVENT_TYPE_RON ? 0 : honba();
                int riichi_ = LastEvent().type() == mjproto::EVENT_TYPE_RON ? 0 : riichi();
                if (ten_move > 0) ten_move += riichi_ * 1000 + honba_ * 300;
                else if (ten_move < 0) ten_move = - (ten_ / 2);
                if (who == pao.value()) ten_move -= ((ten_ / 2) + honba_ * 300);  // 積み棒はパオが払う。パオがロンされたときに注意
            }
        } else {
            for (auto &[who, ten_move]: ten_moves) {
                // ダブロンは上家取り
                int honba_ = LastEvent().type() == mjproto::EVENT_TYPE_RON ? 0 : honba();
                int riichi_ = LastEvent().type() == mjproto::EVENT_TYPE_RON ? 0 : riichi();
                if (ten_move > 0) ten_move += riichi_ * 1000 + honba_ * 300;
                else if (ten_move < 0) ten_move -= honba_ * 300;
            }
        }
        curr_score_.set_riichi(0);

        // set event
        state_.mutable_event_history()->mutable_events()->Add(Event::CreateRon(winner, tile).proto());

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
            state_.mutable_event_history()->mutable_events()->Add(Event::CreateNoWinner().proto());
            is_round_over_ = true;
        };
        // 九種九牌
        if (IsFirstTurnWithoutOpen() && LastEvent().type() == mjproto::EVENT_TYPE_DRAW) {
            state_.mutable_terminal()->mutable_no_winner()->set_type(mjproto::NO_WINNER_TYPE_KYUUSYU);
            mjproto::TenpaiHand tenpai;
            tenpai.set_who(mjproto::AbsolutePos(LastEvent().who()));
            for (auto tile: hand(LastEvent().who()).ToVectorClosed(true)) tenpai.mutable_closed_tiles()->Add(tile.Id());
            state_.mutable_terminal()->mutable_no_winner()->mutable_tenpais()->Add(std::move(tenpai));
            set_terminal_vals();
            return;
        }
        // 四風子連打
        if (IsFourWinds()) {
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
        if (std::all_of(players_.begin(), players_.end(),
                        [&](const Player& player){ return hand(player.position).IsUnderRiichi(); })) {
            state_.mutable_terminal()->mutable_no_winner()->set_type(mjproto::NO_WINNER_TYPE_FOUR_RIICHI);
            // 聴牌の情報が必要なため, ここでreturnしてはいけない.
        }

        // Handが最後リーチで終わってて、かつ一発が残っていることはないはず（通常流局なら）
        assert(state_.terminal().no_winner().type() != mjproto::NO_WINNER_TYPE_NORMAL ||
                !std::any_of(players_.begin(), players_.end(),
                            [&](const Player& player){ return player.is_ippatsu && hand(player.position).IsUnderRiichi();}));

        // set event
        state_.mutable_event_history()->mutable_events()->Add(Event::CreateNoWinner().proto());

        // set terminal
        std::vector<int> is_tenpai = {0, 0, 0, 0};
        for (int i = 0; i < 4; ++i) {
            auto who = AbsolutePos(i);
            if (three_ronned_player and three_ronned_player.value() == who) continue; // 三家和了でロンされた人の聴牌情報は入れない
            if (auto tenpai_hand = EvalTenpai(who); tenpai_hand) {
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
        if (std::any_of(players_.begin(), players_.end(), [](const Player &p){ return p.has_nm; })) {
            int dealer_ix = ToUType(dealer());
            for (int i = 0; i < 4; ++i) {
                if (player(AbsolutePos(i)).has_nm) {
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

        // 途中流局の場合は連荘
        if(Any(state_.terminal().no_winner().type(),
               {mjproto::NO_WINNER_TYPE_KYUUSYU,
                mjproto::NO_WINNER_TYPE_FOUR_RIICHI,
                mjproto::NO_WINNER_TYPE_THREE_RONS,
                mjproto::NO_WINNER_TYPE_FOUR_KANS,
                mjproto::NO_WINNER_TYPE_FOUR_WINDS})){
            return false;
        }

        auto tens_ = tens();
        for (int i = 0; i < 4; ++i) tens_[i] += 4 - i;  // 同点は起家から順に優先されるので +4, +3, +2, +1 する
        auto top_score = *std::max_element(tens_.begin(), tens_.end());

        // 箱割れ
        bool has_minus_point_player = *std::min_element(tens_.begin(), tens_.end()) < 0;
        if (has_minus_point_player) return true;

        // 東南戦
        if (round() < 7) return false;

        // 北入なし
        bool dealer_win_or_tenpai = (Any(LastEvent().type(), {mjproto::EVENT_TYPE_RON, mjproto::EVENT_TYPE_TSUMO})
                                     && std::any_of(state_.terminal().wins().begin(), state_.terminal().wins().end(), [&](const auto x){ return AbsolutePos(x.who()) == dealer(); })) ||
                                    (LastEvent().type() == mjproto::EVENT_TYPE_NO_WINNER && hand(dealer()).IsTenpai());
        if (round() == 11 && !dealer_win_or_tenpai) return true;

        // トップが3万点必要（供託未収）
        bool top_has_30000 = *std::max_element(tens_.begin(), tens_.end()) >= 30000;
        if (!top_has_30000) return false;

        // オーラストップ親の上がりやめあり
        bool dealer_is_not_top = top_score != tens_[ToUType(dealer())];
        return !(dealer_win_or_tenpai && dealer_is_not_top);
    }

    std::pair<State::HandInfo, WinScore> State::EvalWinHand(AbsolutePos who) const noexcept {
        return {HandInfo{hand(who).ToVectorClosed(true), hand(who).Opens(), hand(who).LastTileAdded()},
                YakuEvaluator::Eval(WinInfo(std::move(win_state_info(who)), hand(who).win_info()))};
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
        if (LastEvent().type() == mjproto::EVENT_TYPE_NO_WINNER) {
            // 途中流局や親テンパイで流局の場合は連荘
            if(Any(state_.terminal().no_winner().type(),
                   {mjproto::NO_WINNER_TYPE_KYUUSYU,
                    mjproto::NO_WINNER_TYPE_FOUR_RIICHI,
                    mjproto::NO_WINNER_TYPE_THREE_RONS,
                    mjproto::NO_WINNER_TYPE_FOUR_KANS,
                    mjproto::NO_WINNER_TYPE_FOUR_WINDS})
                    || hand(dealer()).IsTenpai()) {
                return State(player_ids, seed_, round(), honba() + 1, riichi(), tens());
            } else {
                return State(player_ids, seed_, round() + 1, honba() + 1, riichi(), tens());
            }
        } else {
            if (LastEvent().who() == dealer()) {
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

    bool State::HasLastEvent() const {
        auto event_history = EventHistory();
        return !event_history.empty();
    }
    Event State::LastEvent() const {
        auto event_history = EventHistory();
        assert(!event_history.empty());
        return event_history.back();
    }
    std::vector<Event> State::EventHistory() const {
        std::vector<Event> event_history;
        for (auto event : state_.event_history().events()) {
            event_history.emplace_back(std::move(event));
        }
        return event_history;
    }
    std::optional<Tile> State::TargetTile() const {
        auto event_history = EventHistory();
        std::reverse(event_history.begin(), event_history.end());

        for (const auto& event : event_history) {
            if (event.type() == mjproto::EventType::EVENT_TYPE_DISCARD_FROM_HAND or
                event.type() == mjproto::EventType::EVENT_TYPE_DISCARD_DRAWN_TILE) {
                return event.tile();
            }
            if (event.type() == mjproto::EventType::EVENT_TYPE_KAN_ADDED) {
                return event.open().LastTile();
            }
        }
        return std::nullopt;
    }
    
    bool State::IsFirstTurnWithoutOpen() const {
        for (const auto& event: EventHistory()) {
            switch (event.type()) {
                case mjproto::EVENT_TYPE_CHI:
                case mjproto::EVENT_TYPE_PON:
                case mjproto::EVENT_TYPE_KAN_CLOSED:
                case mjproto::EVENT_TYPE_KAN_OPENED:
                case mjproto::EVENT_TYPE_KAN_ADDED:
                    return false;
                case mjproto::EVENT_TYPE_DISCARD_FROM_HAND:
                case mjproto::EVENT_TYPE_DISCARD_DRAWN_TILE:
                    if (ToSeatWind(event.who(), dealer()) == Wind::kNorth) {
                        return false;
                    }
            }
        }
        return true;
    }
    bool State::IsFourWinds() const {
        std::map<TileType,int> discarded_winds;
        for (const auto& event: EventHistory()) {
            switch (event.type()) {
                case mjproto::EVENT_TYPE_CHI:
                case mjproto::EVENT_TYPE_PON:
                case mjproto::EVENT_TYPE_KAN_CLOSED:
                case mjproto::EVENT_TYPE_KAN_OPENED:
                case mjproto::EVENT_TYPE_KAN_ADDED:
                    return false;
                case mjproto::EVENT_TYPE_DISCARD_FROM_HAND:
                case mjproto::EVENT_TYPE_DISCARD_DRAWN_TILE:
                    if (!Is(event.tile().Type(), TileSetType::kWinds)) {
                        return false;
                    }
                    ++discarded_winds[event.tile().Type()];
                    if (discarded_winds.size() > 1) {
                        return false;
                    }
            }
        }
        return discarded_winds.size() == 1 and discarded_winds.begin()->second == 4;
    }


    std::unordered_map<PlayerId, Observation> State::CreateStealAndRonObservation() const {
        std::unordered_map<PlayerId, Observation> observations;
        auto discarder = LastEvent().who();
        auto tile = LastEvent().type() != mjproto::EVENT_TYPE_KAN_ADDED ? LastEvent().tile() : LastEvent().open().LastTile();
        auto has_draw_left = wall_.HasDrawLeft();

        for (int i = 0; i < 4; ++i) {
            auto stealer = AbsolutePos(i);
            if (stealer == discarder) continue;
             auto observation = Observation(stealer, state_);

             // check ron
             if (hand(stealer).IsCompleted(tile) &&
                 CanRon(stealer, tile)) {
                 observation.add_possible_action(Action::CreateRon(stealer));
             }

             // check chi, pon and kan_opened
             if (has_draw_left && LastEvent().type() != mjproto::EVENT_TYPE_KAN_ADDED && !IsFourKanNoWinner()) {  // if 槍槓 or 四槓散了直前の捨て牌, only ron
                auto relative_pos = ToRelativePos(stealer, discarder);
                auto possible_opens = hand(stealer).PossibleOpensAfterOthersDiscard(tile, relative_pos);
                for (const auto & possible_open: possible_opens)
                    observation.add_possible_action(Action::CreateOpen(stealer, possible_open));
             }

             if (!observation.has_possible_action()) continue;
             observation.add_possible_action(Action::CreateNo(stealer));

             observations[player(stealer).player_id] = std::move(observation);
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
                player(who).is_ippatsu,
                IsFirstTurnWithoutOpen() && LastEvent().who() == who
                        && (Any(LastEvent().type(), {mjproto::EVENT_TYPE_DRAW, mjproto::EVENT_TYPE_TSUMO})),
                seat_wind == Wind::kEast,
                is_robbing_kan,
                wall_.dora_count(),
                wall_.ura_dora_count());
        return win_state_info;
    }

    void State::Update(std::vector<Action> &&action_candidates) {
        static_assert(mjproto::ACTION_TYPE_NO < mjproto::ACTION_TYPE_CHI);
        static_assert(mjproto::ACTION_TYPE_CHI < mjproto::ACTION_TYPE_PON);
        static_assert(mjproto::ACTION_TYPE_CHI < mjproto::ACTION_TYPE_KAN_OPENED);
        static_assert(mjproto::ACTION_TYPE_PON < mjproto::ACTION_TYPE_RON);
        static_assert(mjproto::ACTION_TYPE_KAN_OPENED < mjproto::ACTION_TYPE_RON);
        assert(!action_candidates.empty() && action_candidates.size() <= 3);

        if (action_candidates.size() == 1) {
            Update(std::move(action_candidates.front()));
        } else {
            // sort in order Ron > KanOpened > Pon > Chi > No
            std::sort(action_candidates.begin(), action_candidates.end(),
                    [](const Action &x, const Action &y){ return x.type() > y.type(); });
            bool has_ron = action_candidates.front().type() == mjproto::ACTION_TYPE_RON;
            if (has_ron) {
                // ron以外の行動は取られないので消していく
                while (action_candidates.back().type() != mjproto::ACTION_TYPE_RON) action_candidates.pop_back();
                // 上家から順にsortする（ダブロン時に供託が上家取り）
                AbsolutePos from_who = LastEvent().who();
                std::sort(action_candidates.begin(), action_candidates.end(),
                          [&from_who](const Action &x, const Action &y){ return ((ToUType(x.who()) - ToUType(from_who) + 4) % 4) < ((ToUType(y.who()) - ToUType(from_who) + 4) % 4); });
                int ron_count = action_candidates.size();
                if (ron_count == 3) {
                    // 三家和了
                    std::vector<int> ron = {0, 0, 0, 0};
                    for (const auto &action : action_candidates) {
                        if (action.type() == mjproto::ACTION_TYPE_RON) ron[ToUType(action.who())] = 1;
                    }
                    assert(std::accumulate(ron.begin(), ron.end(), 0) == 3);
                    for (int i = 0; i < 4; ++i) {
                        if (ron[i] == 0) three_ronned_player = AbsolutePos(i);
                    }
                    NoWinner();
                    return;
                }
                for (auto &action: action_candidates) {
                    if (action.type() != mjproto::ACTION_TYPE_RON) break;
                    Update(std::move(action));
                }
            } else {
                assert(Any(action_candidates.front().type(), {
                    mjproto::ACTION_TYPE_NO, mjproto::ACTION_TYPE_CHI,
                    mjproto::ACTION_TYPE_PON, mjproto::ACTION_TYPE_KAN_OPENED}));
                Update(std::move(action_candidates.front()));
            }
        }
    }

    void State::Update(Action &&action) {
        assert(Any(LastEvent().type(), {mjproto::EVENT_TYPE_DRAW, mjproto::EVENT_TYPE_DISCARD_FROM_HAND,
                                        mjproto::EVENT_TYPE_DISCARD_DRAWN_TILE, mjproto::EVENT_TYPE_RIICHI,
                                        mjproto::EVENT_TYPE_CHI, mjproto::EVENT_TYPE_PON,
                                        mjproto::EVENT_TYPE_KAN_ADDED, mjproto::EVENT_TYPE_RON}));
        auto who = action.who();
        switch (action.type()) {
            case mjproto::ACTION_TYPE_DISCARD:
                {
                    assert(Any(LastEvent().type(), {mjproto::EVENT_TYPE_DRAW, mjproto::EVENT_TYPE_CHI,
                                                    mjproto::EVENT_TYPE_PON, mjproto::EVENT_TYPE_RON,
                                                    mjproto::EVENT_TYPE_RIICHI}));
                    assert(LastEvent().type() == mjproto::EVENT_TYPE_RIICHI || Any(hand(who).PossibleDiscards(),
                            [&action](Tile possible_discard){ return possible_discard.Equals(action.discard()); }));
                    assert(LastEvent().type() != mjproto::EVENT_TYPE_RIICHI || Any(hand(who).PossibleDiscardsJustAfterRiichi(),
                            [&action](Tile possible_discard){ return possible_discard.Equals(action.discard()); }));
                    assert(require_kan_dora_ <= 1);
                    if (require_kan_dora_) AddNewDora();
                    Discard(who, action.discard());
                    if (IsFourWinds()) {  // 四風子連打
                        NoWinner();
                        return;
                    }
                    // TODO: CreateStealAndRonObservationが2回stateが変わらないのに呼ばれている（CreateObservation内で）
                    if (bool has_steal_or_ron = !CreateStealAndRonObservation().empty(); has_steal_or_ron) return;

                    // 鳴きやロンの候補がなく, 全員が立直していたら四家立直で流局
                    if (std::all_of(players_.begin(), players_.end(),
                                    [&](const Player& player){ return hand(player.position).IsUnderRiichi(); })) {
                        RiichiScoreChange();
                        NoWinner();
                        return;
                    }
                    // 鳴きやロンの候補がなく, 2人以上が合計4つ槓をしていたら四槓散了で流局
                    {
                        std::vector<int> kans;
                        for (const Player& p : players_) {
                            if (int num = hand(p.position).TotalKans(); num) kans.emplace_back(num);
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
            case mjproto::ACTION_TYPE_RIICHI:
                assert(Any(LastEvent().type(), {mjproto::EVENT_TYPE_DRAW}));
                Riichi(who);
                return;
            case mjproto::ACTION_TYPE_TSUMO:
                assert(Any(LastEvent().type(), {mjproto::EVENT_TYPE_DRAW}));
                Tsumo(who);
                return;
            case mjproto::ACTION_TYPE_RON:
                assert(Any(LastEvent().type(), {
                    mjproto::EVENT_TYPE_DISCARD_FROM_HAND, mjproto::EVENT_TYPE_DISCARD_DRAWN_TILE,
                    mjproto::EVENT_TYPE_KAN_ADDED, mjproto::EVENT_TYPE_RON}));
                Ron(who);
                return;
            case mjproto::ACTION_TYPE_CHI:
            case mjproto::ACTION_TYPE_PON:
                assert(Any(LastEvent().type(), {mjproto::EVENT_TYPE_DISCARD_FROM_HAND, mjproto::EVENT_TYPE_DISCARD_DRAWN_TILE}));
                if (require_riichi_score_change_) RiichiScoreChange();
                ApplyOpen(who, action.open());
                return;
            case mjproto::ACTION_TYPE_KAN_OPENED:
                assert(Any(LastEvent().type(), {mjproto::EVENT_TYPE_DISCARD_FROM_HAND, mjproto::EVENT_TYPE_DISCARD_DRAWN_TILE}));
                if (require_riichi_score_change_) RiichiScoreChange();
                ApplyOpen(who, action.open());
                Draw(who);
                return;
            case mjproto::ACTION_TYPE_KAN_CLOSED:
                assert(Any(LastEvent().type(), {mjproto::EVENT_TYPE_DRAW}));
                ApplyOpen(who, action.open());
                // 天鳳のカンの仕様については https://github.com/sotetsuk/mahjong/issues/199 で調べている
                // 暗槓の分で最低一回は新ドラがめくられる
                assert(require_kan_dora_ <= 2);
                while(require_kan_dora_) AddNewDora();
                Draw(who);
                return;
            case mjproto::ACTION_TYPE_KAN_ADDED:
                assert(Any(LastEvent().type(), {mjproto::EVENT_TYPE_DRAW}));
                ApplyOpen(who, action.open());
                // TODO: CreateStealAndRonObservationが状態変化がないのに2回計算されている
                if (auto has_no_ron = CreateStealAndRonObservation().empty(); has_no_ron) {
                    assert(require_kan_dora_ <= 2);
                    while(require_kan_dora_ > 1) AddNewDora();  // 前のカンの分の新ドラをめくる。1回分はここでの加槓の分なので、ここではめくられない
                    Draw(who);
                }
                return;
            case mjproto::ACTION_TYPE_NO:
                assert(Any(LastEvent().type(), {
                    mjproto::EVENT_TYPE_DISCARD_DRAWN_TILE, mjproto::EVENT_TYPE_DISCARD_FROM_HAND,
                    mjproto::EVENT_TYPE_KAN_ADDED}));

                // 加槓のあとに mjproto::ActionType::kNo が渡されるのは槍槓のロンを否定した場合のみ
                if (LastEvent().type() == mjproto::EVENT_TYPE_KAN_ADDED) {
                    Draw(AbsolutePos((ToUType(LastEvent().who()))));  // 嶺上ツモ
                    return;
                }

                // 全員が立直している状態で mjproto::ActionType::kNo が渡されるのは,
                // 4人目に立直した人の立直宣言牌を他家がロンできるけど無視したときのみ.
                // 四家立直で流局とする.
                if (std::all_of(players_.begin(), players_.end(),
                                [&](const Player& player){ return hand(player.position).IsUnderRiichi(); })) {
                    RiichiScoreChange();
                    NoWinner();
                    return;
                }

                // 2人以上が合計4つ槓をしている状態で mjproto::ActionType::kNo が渡されるのは,
                // 4つ目の槓をした人の打牌を他家がロンできるけど無視したときのみ.
                // 四槓散了で流局とする.
                if (IsFourKanNoWinner()) {
                    NoWinner();
                    return;
                }

                if (wall_.HasDrawLeft()) {
                    if (require_riichi_score_change_) RiichiScoreChange();
                    Draw(AbsolutePos((ToUType(LastEvent().who()) + 1) % 4));
                } else {
                    NoWinner();
                }
                return;
            case mjproto::ACTION_TYPE_KYUSYU:
                assert(Any(LastEvent().type(), {mjproto::EVENT_TYPE_DRAW}));
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
            if (int num = hand(p.position).TotalKans(); num) kans.emplace_back(num);
        }
        return std::accumulate(kans.begin(), kans.end(), 0) == 4 and kans.size() > 1;
    }

    mjproto::State State::proto() const {
        return state_;
    }

    std::optional<AbsolutePos> State::HasPao(AbsolutePos winner) const noexcept {
        auto pao = player(winner).hand.HasPao();
        if (pao) return AbsolutePos((ToUType(winner) + ToUType(pao.value())) % 4);
        else return std::nullopt;
    }


    bool State::Equals(const State &other) const noexcept {
        auto seq_eq = [](const auto &x, const auto &y) {
            if (x.size() != y.size()) return false;
            return std::equal(x.begin(), x.end(), y.begin());
        };
        auto tiles_eq = [](const auto &x, const auto &y) {
            if (x.size() != y.size()) return false;
            for (int i = 0; i < x.size(); ++i) if (!Tile(x[i]).Equals(Tile(y[i]))) return false;
            return true;
        };
        auto opens_eq = [](const auto &x, const auto &y) {
            if (x.size() != y.size()) return false;
            for (int i = 0; i < x.size(); ++i) if (!Open(x[i]).Equals(Open(y[i]))) return false;
            return true;
        };
        if (!seq_eq(state_.player_ids(),other.state_.player_ids())) return false;
        if (!google::protobuf::util::MessageDifferencer::Equals(state_.init_score(), other.state_.init_score())) return false;
        if (!tiles_eq(state_.wall(), other.state_.wall())) return false;
        if (!tiles_eq(state_.doras(), other.state_.doras())) return false;
        if (!tiles_eq(state_.ura_doras(), other.state_.ura_doras())) return false;
        for (int i = 0; i < 4; ++i) if (!tiles_eq(state_.private_infos(i).init_hand(), other.state_.private_infos(i).init_hand())) return false;
        for (int i = 0; i < 4; ++i) if (!tiles_eq(state_.private_infos(i).draws(), other.state_.private_infos(i).draws())) return false;
        // EventHistory
        if (state_.event_history().events_size() != other.state_.event_history().events_size()) return false;
        for (int i = 0; i < state_.event_history().events_size(); ++i) {
            const auto &event = state_.event_history().events(i);
            const auto &other_event = other.state_.event_history().events(i);
            if (event.type() != other_event.type()) return false;
            if (event.who() != other_event.who()) return false;
            if (event.tile() != other_event.tile() && !Tile(event.tile()).Equals(Tile(other_event.tile()))) return false;
            if (event.open() != other_event.open() && !Open(event.open()).Equals(Open(other_event.open()))) return false;
        }
        // Terminal
        if (!state_.has_terminal() && !other.state_.has_terminal()) return true;
        if (!google::protobuf::util::MessageDifferencer::Equals(state_.terminal().final_score(), other.state_.terminal().final_score())) return false;
        if (state_.terminal().wins_size() != other.state_.terminal().wins_size()) return false;
        for (int i = 0; i < state_.terminal().wins_size(); ++i) {
            const auto &win = state_.terminal().wins(i);
            const auto &other_win = other.state_.terminal().wins(i);
            if (win.who() != other_win.who()) return false;
            if (win.from_who() != other_win.from_who()) return false;
            if (!tiles_eq(win.closed_tiles(), other_win.closed_tiles())) return false;
            if (!opens_eq(win.opens(), other_win.opens())) return false;
            if (!Tile(win.win_tile()).Equals(Tile(other_win.win_tile()))) return false;
            if (win.fu() != other_win.fu()) return false;
            if (win.ten() != other_win.ten()) return false;
            if (!seq_eq(win.ten_changes(), other_win.ten_changes())) return false;
            if (!seq_eq(win.yakus(), other_win.yakus())) return false;
            if (!seq_eq(win.fans(), other_win.fans())) return false;
            if (!seq_eq(win.yakumans(), other_win.yakumans())) return false;
        }
        const auto &no_winner = state_.terminal().no_winner();
        const auto &other_no_winner = other.state_.terminal().no_winner();
        if (no_winner.tenpais_size() != other_no_winner.tenpais_size()) return false;
        for (int i = 0; i < no_winner.tenpais_size(); ++i) {
            const auto &tenpai = no_winner.tenpais(i);
            const auto &other_tenpai = other_no_winner.tenpais(i);
            if (tenpai.who() != other_tenpai.who()) return false;
            if (!tiles_eq(tenpai.closed_tiles(), other_tenpai.closed_tiles())) return false;
        }
        if (!seq_eq(no_winner.ten_changes(), other_no_winner.ten_changes())) return false;
        if (no_winner.type() != other_no_winner.type()) return false;
        if (state_.terminal().is_game_over() != other.state_.terminal().is_game_over()) return false;
        return true;
    }

    bool State::CanReach(const State &other) const noexcept {
        auto seq_eq = [](const auto &x, const auto &y) {
            if (x.size() != y.size()) return false;
            return std::equal(x.begin(), x.end(), y.begin());
        };
        auto tiles_eq = [](const auto &x, const auto &y) {
            if (x.size() != y.size()) return false;
            for (int i = 0; i < x.size(); ++i) if (!Tile(x[i]).Equals(Tile(y[i]))) return false;
            return true;
        };

        if (this->Equals(other)) return true;

        // いくつかの初期状態が同じである必要がある
        if (!seq_eq(state_.player_ids(),other.state_.player_ids())) return false;
        if (!google::protobuf::util::MessageDifferencer::Equals(state_.init_score(), other.state_.init_score())) return false;
        if (!tiles_eq(state_.wall(), other.state_.wall())) return false;

        // 現在の時点まではイベントがすべて同じである必要がある
        if (state_.event_history().events_size() >= other.state_.event_history().events_size()) return false;  // イベント長が同じならそもそもEqualのはず
        for (int i = 0; i < state_.event_history().events_size(); ++i) {
            const auto &event = state_.event_history().events(i);
            const auto &other_event = other.state_.event_history().events(i);
            if (event.type() != other_event.type()) return false;
            if (event.who() != other_event.who()) return false;
            if (event.tile() != other_event.tile() && !Tile(event.tile()).Equals(Tile(other_event.tile()))) return false;
            if (event.open() != other_event.open() && !Open(event.open()).Equals(Open(other_event.open()))) return false;
        }

        // Drawがすべて現時点までは同じである必要がある (配牌は山が同じ時点で同じ）
        for (int i = 0; i < 4; ++i) {
            const auto &draws = state_.private_infos(i).draws();
            const auto &other_draws = other.state_.private_infos(i).draws();
            if (draws.size() > other_draws.size()) return false;
            for (int j = 0; j < draws.size(); ++j) if (!Tile(draws[j]).Equals(Tile(other_draws[j]))) return false;
        }

        // もしゲーム終了しているなら、Equalでない時点でダメ
        return !IsRoundOver();
    }

    // #398 追加分
    // action validators
    bool State::CanRon(AbsolutePos who, Tile tile) const {
        // フリテンでないことを確認
        if ((player(who).machi & player(who).discards).any()) return false;
        if ((player(who).machi & player(who).missed_tiles).any()) return false;
        return YakuEvaluator::CanWin(WinInfo(std::move(win_state_info(who)), hand(who).win_info()).Ron(tile));
    }

    bool State::CanRiichi(AbsolutePos who) const {
        if (hand(who).IsUnderRiichi()) return false;
        if (!wall_.HasNextDrawLeft()) return false;
        return hand(who).CanRiichi(ten(who));
    }

    bool State::CanTsumo(AbsolutePos who) const {
        return YakuEvaluator::CanWin(WinInfo(std::move(win_state_info(who)), hand(who).win_info()));
    }

    std::optional<State::HandInfo> State::EvalTenpai(AbsolutePos who) const noexcept {
        if (!hand(who).IsTenpai()) return std::nullopt;
        return HandInfo{hand(who).ToVectorClosed(true), hand(who).Opens(), hand(who).LastTileAdded()};
    }
}  // namespace mj
