#ifndef MAHJONG_PLAYER_H
#define MAHJONG_PLAYER_H

#include "hand.h"
#include "river.h"
#include <mahjong.grpc.pb.h>

namespace mj
{
    class Player
    {
    public:
        Player() = default;
        Player(AbsolutePos position, River river, Hand initial_hand);
        [[nodiscard]] AbsolutePos position() const;
        [[nodiscard]] const Hand& hand() const;
        [[nodiscard]] const River& river() const;

        // action validators
        std::vector<Tile> PossibleDiscards() const;  // TODO(sotetsuk): Current implementation has the tiles with same type (e.g., 2m x 3). What is the Tenhou's implementation? Only first id? or any id?
        std::vector<Tile> PossibleDiscardsAfterRiichi();
        std::vector<Open> PossibleOpensAfterOthersDiscard(Tile tile, RelativePos from) const;  // includes Chi, Pon, and KanOpened
        std::vector<Open> PossibleOpensAfterDraw();  // includes KanClosed and KanAdded
        bool CanRon(Tile tile) const;  // This does not take furiten and fan into account.
        bool IsCompleted();
        bool CanRiichi();
        //bool CanNineTiles(bool IsDealer);  // 九種九牌

        // apply actions
        void Draw(Tile tile);
        void Riichi();  // After riichi, hand is fixed
        void ApplyOpen(Open open);  // TODO: (sotetsuk) current implementation switch private method depending on OpenType. This is not smart way to do dynamic polymorphism.
        void Ron(Tile tile);
        void RonAfterOthersKan(Tile tile);
        void Tsumo();  // should be called after draw like h.Draw(tile); if (h.IsCompleted(w)) h.Tsumo();
        Tile Discard(Tile tile);

        // get winning info
        WinningScore EvalScore() const noexcept ;

        // river
        void Discard(Tile tile, bool tsumogiri);
        Tile latest_discard() const;
    private:
        friend class Observation;  // refers to initial_hand_
        AbsolutePos position_;
        River river_;
        Hand hand_;
        mjproto::InitialHand initial_hand_;
    };
}  // namespace mj

#endif //MAHJONG_PLAYER_H
