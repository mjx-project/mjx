#ifndef MAHJONG_HAND_H
#define MAHJONG_HAND_H

#include <cstdint>
#include <optional>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "mjx/internal/open.h"
#include "mjx/internal/tile.h"
#include "mjx/internal/win_cache.h"
#include "mjx/internal/win_info.h"
#include "mjx/internal/win_score.h"

namespace mjx::internal {
class HandParams;
class Hand {
 public:
  Hand() = default;
  explicit Hand(std::vector<Tile> tiles);
  Hand(std::vector<Tile>::iterator begin, std::vector<Tile>::iterator end);
  Hand(std::vector<Tile>::const_iterator begin,
       std::vector<Tile>::const_iterator end);
  /*
   * Utility constructor only for test usage. This simplifies Chi/Pon/Kan
   * information:
   *   - Tile ids are successive and always zero-indexed
   *   - Chi: Stolen tile is always the smallest one. E.g., [1m]2m3m
   *   - Pon: Stolen tile id is always zero. Stolen player is always left
   * player.
   *   - Kan: Stolen tile id is always zero. Stolen player is always left
   * player. (Last tile of KanAdded has last id = 3)
   *
   *  Usage:
   *    auto hand = Hand(
   *        HandParams("m1,m2,m3,m4,m5,rd,rd").Chi("m7,m8,m9").KanAdded("p1,p1,p1,p1")
   *    );
   *
   */
  explicit Hand(const HandParams &hand_params);

  // accessor to hand internal state
  [[nodiscard]] HandStage stage() const;
  [[nodiscard]] std::optional<Tile> LastTileAdded() const;
  [[nodiscard]] bool IsMenzen() const;
  bool IsUnderRiichi() const;
  bool IsDoubleRiichi() const;
  [[nodiscard]] std::size_t Size() const;
  [[nodiscard]] std::size_t SizeOpened() const;
  [[nodiscard]] std::size_t SizeClosed() const;
  [[nodiscard]] std::vector<Tile> ToVector(bool sorted = false) const;
  [[nodiscard]] std::vector<Tile> ToVectorClosed(bool sorted = false) const;
  [[nodiscard]] std::vector<Tile> ToVectorOpened(bool sorted = false) const;
  std::array<std::uint8_t, 34> ToArray();
  std::array<std::uint8_t, 34> ToArrayClosed();
  std::array<std::uint8_t, 34> ToArrayOpened();
  [[nodiscard]] std::vector<Open> Opens()
      const;  // TODO(sotetsuk): Should we avoid raw pointer?
  [[nodiscard]] std::string ToString(bool verbose = false) const;
  [[nodiscard]] TileTypeCount ClosedTileTypes() const noexcept;
  [[nodiscard]] TileTypeCount AllTileTypes() const noexcept;
  int TotalKans() const noexcept;  // 四槓散了の判定に使用する.
  std::optional<RelativePos> HasPao() const noexcept;

  // action validators
  std::vector<std::pair<Tile, bool>> PossibleDiscards()
      const;  // 同じ種類（タイプ）の牌については、idが一番小さいものだけを返す。赤とツモ切り牌だけ例外。
  std::vector<std::pair<Tile, bool>> PossibleDiscardsToTakeTenpai()
      const;  // 同上
  std::vector<std::pair<Tile, bool>> PossibleDiscardsJustAfterRiichi()
      const;  // 同上
  std::vector<Open> PossibleOpensAfterOthersDiscard(
      Tile tile, RelativePos from) const;  // includes Chi, Pon, and KanOpened
  std::vector<Open> PossibleOpensAfterDraw()
      const;  // includes KanClosed and KanAdded
  bool IsCompleted() const;
  bool IsCompleted(Tile additional_tile) const;
  bool CanRiichi(std::int32_t ten = 25000) const;  // デフォルト25000点
  bool CanTakeTenpai() const;
  bool IsTenpai() const;
  bool CanNineTiles()
      const;  // 九種九牌。一巡目かどうか等はState内で判断する。ここでは牌が九種九牌かどうかのみ

  // apply actions
  void Draw(Tile tile);
  void Riichi(bool double_riichi = false);  // After riichi, hand is fixed
  void ApplyOpen(Open open);  // TODO: (sotetsuk) current implementation switch
                              // private method depending on OpenType. This is
                              // not smart way to do dynamic polymorphism.
  void Ron(Tile tile);
  void Tsumo();  // should be called after draw like h.Draw(tile); if
                 // (h.IsCompleted(w)) h.Tsumo();
  std::pair<Tile, bool> Discard(Tile tile);

  // get winning info
  [[nodiscard]] WinHandInfo win_info() const noexcept;

  mjxproto::Hand ToProto() const noexcept;

  // operators
  bool operator==(const Hand &right) const noexcept;
  bool operator!=(const Hand &right) const noexcept;

 private:
  std::unordered_set<Tile, HashTile> closed_tiles_;
  std::vector<Open> opens_;  // Though open only uses 16 bits, to handle
                             // different open types, we need to use pointer
  std::unordered_set<Tile, HashTile> undiscardable_tiles_;
  std::optional<Tile> last_tile_added_;
  HandStage stage_;
  bool under_riichi_ = false;
  bool double_riichi_ = false;

  // possible actions
  std::vector<Open> PossibleChis(
      Tile tile) const;  // E.g., 2m 3m [4m] vs 3m [4m] 5m
  std::vector<Open> PossiblePons(Tile tile, RelativePos from)
      const;  // E.g., with red or not  TODO: check the id choice strategy of
              // tenhou (smalelr one) when it has 2 identical choices.
  std::vector<Open> PossibleKanOpened(Tile tile, RelativePos from) const;
  std::vector<Open> PossibleKanClosed()
      const;  // TODO: which tile id should be used to represent farleft left
              // bits? (current is type * 4 + 0)
  std::vector<Open> PossibleKanAdded() const;
  void ApplyKanAdded(Open open);

  // apply actions
  void ApplyChi(Open open);
  void ApplyPon(Open open);
  void ApplyKanOpened(Open open);
  void ApplyKanClosed(Open open);

  // utils
  static bool IsTenpai(const TileTypeCount &closed_tile_types);
  std::vector<std::pair<Tile, bool>> AllPossibleDiscards() const;
  // 鳴いた後に捨てる牌がある鳴きだけを選ぶ
  std::vector<Open> SelectDiscardableOpens(
      const std::vector<Open> &opens) const;

  explicit Hand(std::vector<std::string> closed,
                std::vector<std::vector<std::string>> chis = {},
                std::vector<std::vector<std::string>> pons = {},
                std::vector<std::vector<std::string>> kan_openeds = {},
                std::vector<std::vector<std::string>> kan_closeds = {},
                std::vector<std::vector<std::string>> kan_addeds = {},
                std::string tsumo = "", std::string ron = "",
                bool riichi = false, bool after_kan = false);
};

class HandParams {
 public:
  // Usage:
  //   auto h =
  //   Hand(HandParams("m1,m1,wd,wd").Chi("m2,m3,m4").Pon("m9,m9,m9").Pon("rd,rd,rd").Tsumo("wd"));
  explicit HandParams(const std::string &closed);
  HandParams &Chi(const std::string &chi);
  HandParams &Pon(const std::string &pon);
  HandParams &KanOpened(const std::string &kan_opened);
  HandParams &KanClosed(const std::string &kan_closed);
  HandParams &KanAdded(const std::string &kan_added);
  HandParams &Riichi();
  HandParams &Tsumo(const std::string &tsumo, bool after_kan = false);
  HandParams &Ron(const std::string &ron, bool after_kan = false);

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
  void Push(const std::string &input,
            std::vector<std::vector<std::string>> &vec);
};
}  // namespace mjx::internal

#endif  // MAHJONG_HAND_H
