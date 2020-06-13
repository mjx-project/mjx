#ifndef MAHJONG_HAND_H
#define MAHJONG_HAND_H

#include <cstdint>
#include <unordered_set>
#include <set>
#include <vector>

#include <tile.h>
#include <unordered_map>
#include "open.h"
#include "win_cache.h"

namespace mj
{
    class Hand
    {
    public:
        explicit Hand(const std::vector<TileId> &vector);
        explicit Hand(const std::vector<TileType> &vector);
        explicit Hand(std::vector<Tile> tiles);
        Hand(std::vector<Tile>::iterator begin, std::vector<Tile>::iterator end);
        /*
         * Utility constructor only for test usage. This simplifies Chi/Pon/Kan information:
         *   - Tile ids are successive and always zero-indexed
         *   - Chi: Stolen tile is always the smallest one. E.g., [1m]2m3m
         *   - Pon: Stolen tile id is always zero. Stolen player is always left player.
         *   - Kan: Stolen tile id is always zero. Stolen player is always left player. (Last tile of KanAdded has last id = 3)
         *
         *  Usage:
         *    auto hand = Hand(
         *            {"m4", "m5", "m6", "rd", "rd"},  // closed
         *            {{"m1", "m2", "m3"}, {"m7", "m8", "m9"}},  // chi
         *            {},  // pon
         *            {},  // kan_opend
         *            {},  // kan_closed
         *            {{"p1", "p1", "p1", "p1"}}  // kan_added
         *    );
         */
        Hand(std::vector<std::string> closed,
             std::vector<std::vector<std::string>> chis = {},
             std::vector<std::vector<std::string>> pons = {},
             std::vector<std::vector<std::string>> kan_openeds = {},
             std::vector<std::vector<std::string>> kan_closeds = {},
             std::vector<std::vector<std::string>> kan_addeds = {});

        // accessor to hand internal state
        HandStage Stage();
        std::optional<Tile> LastTileAdded();
        std::optional<ActionType> LastActionType();
        bool IsMenzen();
        bool IsUnderRiichi();
        std::size_t Size();
        std::size_t SizeOpened();
        std::size_t SizeClosed();
        std::size_t SizeChi();
        std::size_t SizePon();
        std::size_t SizeKanOpened();
        std::size_t SizeKanClosed();
        std::size_t SizeAdded();
        std::size_t NumChi();
        std::size_t NumPon();
        std::size_t NumKanOpened();
        std::size_t NumKanClosed();
        std::size_t NumKanAdded();
        std::vector<Tile> ToVector(bool sorted = false);
        std::vector<Tile> ToVectorClosed(bool sorted = false);
        std::vector<Tile> ToVectorOpened(bool sorted = false);
        std::array<std::uint8_t, 34> ToArray();
        std::array<std::uint8_t, 34> ToArrayClosed();
        std::array<std::uint8_t, 34> ToArrayOpened();

        // action validators
        std::vector<Tile> PossibleDiscards();
        std::vector<std::unique_ptr<Open>> PossibleOpensAfterOthersDiscard(Tile tile, RelativePos from);  // includes Chi, Pon, and KanOpened
        std::vector<std::unique_ptr<Open>> PossibleOpensAfterDraw();  // includes KanClosed and KanAdded
        bool CanComplete(Tile tile, const WinningHandCache &win_cache);  // This does not take furiten and fan into account.
        bool CanRiichi(const WinningHandCache &win_cache);
        bool CanNineTiles(bool IsDealer);  // 九種九牌

        // apply actions
        void Draw(Tile tile);
        void Riichi();  // Fixes hand
        void ApplyChi(std::unique_ptr<Open> open);
        void ApplyPon(std::unique_ptr<Open> open);
        void ApplyKanOpened(std::unique_ptr<Open> open);
        void ApplyKanClosed(std::unique_ptr<Open> open);
        void ApplyKanAdded(std::unique_ptr<Open> open);
        void Ron(Tile tile);
        void Tsumo(Tile tile);
        Tile Discard(Tile tile);

        // utility
        bool Has(const std::vector<TileType> &tiles);
  private:
        std::unordered_set<Tile, HashTile> closed_tiles_;
        std::set<std::unique_ptr<Open>> open_sets_;  // Though open only uses 16 bits, to handle different open types, we need to use pointer
        std::unordered_set<Tile, HashTile> undiscardable_tiles_;
        std::optional<Tile> last_tile_added_;
        std::optional<ActionType> last_action_type_;
        HandStage hand_phase_;
        bool under_riichi_;

        // possible actions
        std::vector<std::unique_ptr<Open>> PossibleChis(Tile tile);  // E.g., 2m 3m [4m] vs 3m [4m] 5m
        std::vector<std::unique_ptr<Open>> PossiblePons(Tile tile, RelativePos from);  // E.g., with red or not  TODO: check the id choice strategy of tenhou (smalelr one) when it has 2 identical choices.
        std::vector<std::unique_ptr<Open>> PossibleKanOpened(Tile tile, RelativePos from);
        std::vector<std::unique_ptr<Open>> PossibleKanClosed();  // TODO: which tile id should be used to represent farleft left bits? (current is type * 4 + 0)
        std::vector<std::unique_ptr<Open>> PossibleKanAdded();
   };
}  // namespace mj

#endif //MAHJONG_HAND_H
