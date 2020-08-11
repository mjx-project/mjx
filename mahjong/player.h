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
        Player(PlayerId player_id, AbsolutePos position, River river, Hand initial_hand);
        [[nodiscard]] AbsolutePos position() const;

        // action validators
        std::vector<Tile> PossibleDiscards() const;  // TODO(sotetsuk): Current implementation has the tiles with same type (e.g., 2m x 3). What is the Tenhou's implementation? Only first id? or any id?
        std::vector<Tile> PossibleDiscardsAfterRiichi() const;
        std::vector<Open> PossibleOpensAfterOthersDiscard(Tile tile, RelativePos from) const;  // includes Chi, Pon, and KanOpened
        std::vector<Open> PossibleOpensAfterDraw() const;  // includes KanClosed and KanAdded
        bool IsCompleted(Tile additional_tile) const;  // This does not take into account yaku and furiten
        bool IsCompleted() const;  // This does not take into account yaku and furiten
        bool CanRon(Tile tile) const;
        bool CanTsumo() const;
        bool CanRiichi() const;
        bool IsTenpai() const;
        //bool CanNineTiles(bool IsDealer);  // 九種九牌

        // apply actions
        void Draw(Tile tile);
        void Riichi();  // After riichi, hand is fixed
        void ApplyOpen(Open open);  // TODO: (sotetsuk) current implementation switch private method depending on OpenType. This is not smart way to do dynamic polymorphism.
        void Ron(Tile tile);
        void RonAfterOthersKan(Tile tile);
        void Tsumo();  // should be called after draw like h.Draw(tile); if (h.IsCompleted(w)) h.Tsumo();
        std::pair<Tile, bool> Discard(Tile tile);  // return whether tsumogiri or not

        // get winning info
        [[nodiscard]] std::pair<HandInfo, WinScore> EvalWinHand(WinStateInfo win_state_info) const noexcept ;
        [[nodiscard]] std::optional<HandInfo> EvalTenpai() const noexcept ;

        // river
        void Discard(Tile tile, bool tsumogiri);
        Tile latest_discard() const;

        [[nodiscard]] PlayerId player_id() const;
    private:
        PlayerId player_id_;
        AbsolutePos position_;
        River river_;
        Hand hand_;
    };
}  // namespace mj

#endif //MAHJONG_PLAYER_H
