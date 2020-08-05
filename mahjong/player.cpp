#include "player.h"

namespace mj
{
    Player::Player(AbsolutePos position, River river, Hand initial_hand):
    position_(position), river_(std::move(river)), hand_(std::move(initial_hand))
    {
        assert(hand_.Stage() == HandStage::kAfterDiscards);
        assert(hand_.Size() == 13);
        assert(hand_.Opens().empty());
    }

    AbsolutePos Player::position() const {
        return position_;
    }

    const Hand &Player::hand() const {
        return hand_;
    }


    const River &Player::river() const {
        return river_;
    }

    // action validators
    std::vector<Tile> Player::PossibleDiscards() const {
        return hand_.PossibleDiscards();
    }

    std::vector<Tile> Player::PossibleDiscardsAfterRiichi() {
        return hand_.PossibleDiscardsAfterRiichi();
    }

    std::vector<Open> Player::PossibleOpensAfterOthersDiscard(Tile tile, RelativePos from) const {
        return hand_.PossibleOpensAfterOthersDiscard(tile, from);
    }

    std::vector<Open> Player::PossibleOpensAfterDraw() {
        return hand_.PossibleOpensAfterDraw();
    }

    bool Player::CanRon(Tile tile) const {
        // TODO: ここでフリテンでないことを確認
        return hand_.CanRon(tile);
    }

    bool Player::IsCompleted() const {
        return hand_.IsCompleted();
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
    WinScore Player::EvalScore(WinStateInfo win_state_info) const noexcept {
        return hand_.EvalScore(win_state_info);
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
}  // namespace mj
