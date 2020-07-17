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
#include "win_info.h"

namespace mj
{
    class HandParams;
    class Hand
    {
    public:
        Hand() = default;
        explicit Hand(std::vector<Tile> tiles);
        Hand(std::vector<Tile>::iterator begin, std::vector<Tile>::iterator end);
        Hand(std::vector<Tile>::const_iterator begin, std::vector<Tile>::const_iterator end);
        /*
         * Utility constructor only for test usage. This simplifies Chi/Pon/Kan information:
         *   - Tile ids are successive and always zero-indexed
         *   - Chi: Stolen tile is always the smallest one. E.g., [1m]2m3m
         *   - Pon: Stolen tile id is always zero. Stolen player is always left player.
         *   - Kan: Stolen tile id is always zero. Stolen player is always left player. (Last tile of KanAdded has last id = 3)
         *
         *  Usage:
         *    auto hand = Hand(
         *        HandParams("m1,m2,m3,m4,m5,rd,rd").Chi("m7,m8,m9").KanAdded("p1,p1,p1,p1")
         *    );
         *
         */
        explicit Hand(const HandParams &hand_params);

        // accessor to hand internal state
        [[nodiscard]] HandStage Stage() const;
        [[nodiscard]] std::optional<Tile> LastTileAdded() const;
        [[nodiscard]] bool IsMenzen() const;
        bool IsUnderRiichi();
        [[nodiscard]] std::size_t Size() const;
        [[nodiscard]] std::size_t SizeOpened() const;
        [[nodiscard]] std::size_t SizeClosed() const;
        [[nodiscard]] std::vector<Tile> ToVector(bool sorted = false) const;
        [[nodiscard]] std::vector<Tile> ToVectorClosed(bool sorted = false) const;
        [[nodiscard]] std::vector<Tile> ToVectorOpened(bool sorted = false) const;
        std::array<std::uint8_t, 34> ToArray();
        std::array<std::uint8_t, 34> ToArrayClosed();
        std::array<std::uint8_t, 34> ToArrayOpened();
        [[nodiscard]] std::vector<const Open*> Opens() const;  // TODO(sotetsuk): Should we avoid raw pointer?
        [[nodiscard]] std::string ToString(bool verbose = false) const;
        [[nodiscard]] TileTypeCount ClosedTileTypes() const noexcept ;
        [[nodiscard]] TileTypeCount AllTileTypes() const noexcept ;

        // action validators
        std::vector<Tile> PossibleDiscards() const;  // TODO(sotetsuk): Current implementation has the tiles with same type (e.g., 2m x 3). What is the Tenhou's implementation? Only first id? or any id?
        std::vector<Tile> PossibleDiscardsAfterRiichi(const WinningHandCache &win_cache);
        std::vector<std::unique_ptr<Open>> PossibleOpensAfterOthersDiscard(Tile tile, RelativePos from) const;  // includes Chi, Pon, and KanOpened
        std::vector<std::unique_ptr<Open>> PossibleOpensAfterDraw();  // includes KanClosed and KanAdded
        bool CanRon(Tile tile) const;  // This does not take furiten and fan into account.
        bool IsCompleted();
        bool CanRiichi();
        bool CanNineTiles(bool IsDealer);  // 九種九牌

        // apply actions
        void Draw(Tile tile);
        void Riichi();  // After riichi, hand is fixed
        void ApplyOpen(std::unique_ptr<Open> open);  // TODO: (sotetsuk) current implementation switch private method depending on OpenType. This is not smart way to do dynamic polymorphism.
        void Ron(Tile tile);
        void RonAfterOthersKan(Tile tile);
        void Tsumo();  // should be called after draw like h.Draw(tile); if (h.IsCompleted(w)) h.Tsumo();
        Tile Discard(Tile tile);

        // get winning info
        WinningInfo ToWinningInfo() const noexcept ;
        WinningInfo ToWinningInfo(const WinningStateInfo& win_state_info) const noexcept ;
    private:
        std::unordered_set<Tile, HashTile> closed_tiles_;
        std::vector<std::unique_ptr<Open>> opens_;  // Though open only uses 16 bits, to handle different open types, we need to use pointer
        std::unordered_set<Tile, HashTile> undiscardable_tiles_;
        std::optional<Tile> last_tile_added_;
        HandStage stage_;
        bool under_riichi_{};

        // possible actions
        std::vector<std::unique_ptr<Open>> PossibleChis(Tile tile) const;  // E.g., 2m 3m [4m] vs 3m [4m] 5m
        std::vector<std::unique_ptr<Open>> PossiblePons(Tile tile, RelativePos from) const;  // E.g., with red or not  TODO: check the id choice strategy of tenhou (smalelr one) when it has 2 identical choices.
        std::vector<std::unique_ptr<Open>> PossibleKanOpened(Tile tile, RelativePos from) const;
        std::vector<std::unique_ptr<Open>> PossibleKanClosed();  // TODO: which tile id should be used to represent farleft left bits? (current is type * 4 + 0)
        std::vector<std::unique_ptr<Open>> PossibleKanAdded();
        void ApplyKanAdded(std::unique_ptr<Open> open);

        // apply actions
        void ApplyChi(std::unique_ptr<Open> open);
        void ApplyPon(std::unique_ptr<Open> open);
        void ApplyKanOpened(std::unique_ptr<Open> open);
        void ApplyKanClosed(std::unique_ptr<Open> open);

        explicit Hand(std::vector<std::string> closed,
             std::vector<std::vector<std::string>> chis = {},
             std::vector<std::vector<std::string>> pons = {},
             std::vector<std::vector<std::string>> kan_openeds = {},
             std::vector<std::vector<std::string>> kan_closeds = {},
             std::vector<std::vector<std::string>> kan_addeds = {},
             std::string tsumo = "", std::string ron = "",
             bool riichi = false, bool after_kan = false);
    };

    class HandParams
    {
    public:
        // Usage:
        //   auto h = Hand(HandParams("m1m1wdwd").Chi("m2m3m4").Pon("m9m9m9").Pon("rdrdrd").Tsumo("wd"));
        explicit HandParams(const std::string &closed);
        HandParams& Chi(const std::string &chi);
        HandParams& Pon(const std::string &pon);
        HandParams& KanOpened(const std::string &kan_opened);
        HandParams& KanClosed(const std::string &kan_closed);
        HandParams& KanAdded(const std::string &kan_added);
        HandParams& Riichi();
        HandParams& Tsumo(const std::string &tsumo, bool after_kan = false);
        HandParams& Ron(const std::string &ron, bool after_kan = false);
    private:
        friend class Hand;
        std::vector<std::string> closed_ = {};
        std::vector<std::vector<std::string>> chis_ = {};
        std::vector<std::vector<std::string>> pons_ = {};
        std::vector<std::vector<std::string>> kan_openeds_ = {};
        std::vector<std::vector<std::string>> kan_closeds_ = {};
        std::vector<std::vector<std::string>> kan_addeds_ = {};
        std::string tsumo_ = "";
        std::string ron_ = "";
        bool after_kan_ = false;
        bool riichi_ = false;
        void Push(const std::string &input, std::vector<std::vector<std::string>> &vec);
    };
}  // namespace mj

#endif //MAHJONG_HAND_H
