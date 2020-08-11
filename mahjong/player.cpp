#include "player.h"

#include <utility>
#include "yaku_evaluator.h"

namespace mj
{
    Player::Player(PlayerId player_id, AbsolutePos position, River river, Hand initial_hand):
    player_id_(std::move(player_id)), position_(position), river_(std::move(river)), hand_(std::move(initial_hand))
    {
        assert(hand_.Stage() == HandStage::kAfterDiscards);
        assert(hand_.Size() == 13);
        assert(hand_.Opens().empty());
    }

    AbsolutePos Player::position() const {
        return position_;
    }

    // action validators
    std::vector<Tile> Player::PossibleDiscards() const {
        return hand_.PossibleDiscards();
    }

    std::vector<Tile> Player::PossibleDiscardsAfterRiichi() const {
        return hand_.PossibleDiscardsAfterRiichi();
    }

    std::vector<Open> Player::PossibleOpensAfterOthersDiscard(Tile tile, RelativePos from) const {
        return hand_.PossibleOpensAfterOthersDiscard(tile, from);
    }

    std::vector<Open> Player::PossibleOpensAfterDraw() const {
        return hand_.PossibleOpensAfterDraw();
    }

    bool Player::CanRon(Tile tile) const {
        // TODO: ここでフリテンでないことを確認
        return YakuEvaluator::CanWin(WinInfo(hand_.win_info()).Ron(tile));
    }

    bool Player::CanRiichi() const {
        // TODO: ツモ番があるかどうかをここで確認
        return hand_.CanRiichi();
    }

    //bool Player::CanNineTiles(bool IsDealer) {
    //    return hand_.CanNineTiles(IsDealer);
    //}

    // apply actions
    void Player::Draw(Tile tile) {
        hand_.Draw(tile);
    }
    void Player::Riichi() {
        hand_.Riichi();
    }

    void Player::ApplyOpen(Open open) {
        hand_.ApplyOpen(open);
    }

    void Player::Ron(Tile tile) {
        hand_.Ron(tile);
    }

    void Player::RonAfterOthersKan(Tile tile) {
        hand_.RonAfterOthersKan(tile);
    }

    void Player::Tsumo() {
        hand_.Tsumo();
    }

    std::pair<Tile, bool> Player::Discard(Tile tile) {
        return hand_.Discard(tile);
    };

    // get winning info
    std::pair<HandInfo, WinScore> Player::EvalWinHand(WinStateInfo win_state_info) const noexcept {
        return {HandInfo{hand_.ToVectorClosed(true), hand_.Opens(), hand_.LastTileAdded()},
                YakuEvaluator::Eval(WinInfo(std::move(win_state_info), hand_.win_info()))};
    }

    // river
    void Player::Discard(Tile tile, bool tsumogiri) {
        river_.Discard(tile, tsumogiri);
    }

    Tile Player::latest_discard() const {
        return river_.latest_discard();
    }

    bool Player::IsTenpai() const {
        return hand_.IsTenpai();
    }

    std::optional<HandInfo> Player::EvalTenpai() const noexcept {
        if (!IsTenpai()) return std::nullopt;
        return HandInfo{hand_.ToVectorClosed(true), hand_.Opens(), hand_.LastTileAdded()};
    }

    PlayerId Player::player_id() const {
        return player_id_;
    }

    bool Player::CanTsumo() const {
        return hand_.IsCompleted();
        // if (!hand_.IsCompleted()) return false;
        // return YakuEvaluator::CanWin(WinInfo(hand_.win_info()));
    }

    bool Player::IsCompleted(Tile additional_tile) const {
        return hand_.IsCompleted(additional_tile);
    }

    bool Player::IsCompleted() const {
        return hand_.IsCompleted();
    }
}  // namespace mj
