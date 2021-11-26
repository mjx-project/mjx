#ifndef MJX_PROJECT_HAND_H
#define MJX_PROJECT_HAND_H

#include "mjx/internal/mjx.grpc.pb.h"

namespace mjx {
class Hand {
 public:
  Hand() = default;
  explicit Hand(mjxproto::Hand proto);
  explicit Hand(const std::string& json);
  bool operator==(const Hand& other) const noexcept;
  bool operator!=(const Hand& other) const noexcept;

  std::string ToJson() const noexcept;
  bool IsTenpai() const;
  int ShantenNumber() const;
  std::vector<int> EffectiveDrawTypes() const;
  std::vector<int> EffectiveDiscardTypes() const;
  std::array<uint8_t, 34> ClosedTileTypes() const noexcept;
  std::vector<int> ClosedTiles() const noexcept;
  std::vector<int> Opens() const;

  const mjxproto::Hand& proto() const noexcept;

 private:
  mjxproto::Hand proto_{};
};
}  // namespace mjx

#endif  // MJX_PROJECT_HAND_H
