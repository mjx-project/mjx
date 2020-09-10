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
        bool CanRon(Tile tile, WinStateInfo &&win_state_info) const;
        bool CanTsumo(WinStateInfo &&win_state_info) const;
        bool CanRiichi() const;
        bool IsTenpai() const;
        bool IsUnderRiichi() const;
        bool CanNineTiles() const;
        int TotalKans() const;

        // apply actions
        void Draw(Tile tile);
        void Riichi(bool double_riichi = false);  // After riichi, hand is fixed
        void ApplyOpen(Open open);  // TODO: (sotetsuk) current implementation switch private method depending on OpenType. This is not smart way to do dynamic polymorphism.
        void Ron(Tile tile);
        void RonAfterOthersKan(Tile tile);
        void Tsumo();  // should be called after draw like h.Draw(tile); if (h.IsCompleted(w)) h.Tsumo();
        std::pair<Tile, bool> Discard(Tile tile);  // return whether tsumogiri or not

        std::vector<Tile> closed_tiles() const ;

        // get winning info
        [[nodiscard]] std::pair<HandInfo, WinScore> EvalWinHand(WinStateInfo &&win_state_info) const noexcept ;
        [[nodiscard]] std::optional<HandInfo> EvalTenpai() const noexcept ;

        // river
        void Discard(Tile tile, bool tsumogiri);
        Tile latest_discard() const;

        [[nodiscard]] PlayerId player_id() const;
    private:
        std::uint64_t machi_ = 0;    // 上がりの形になるための待ち(役の有無を考慮しない). bitで管理する
        std::uint64_t discards_ = 0; // 今までに捨てた牌のset. bitで管理する
        PlayerId player_id_;
        AbsolutePos position_;
        River river_;
        Hand hand_;
    };
}  // namespace mj

#endif //MAHJONG_PLAYER_H
