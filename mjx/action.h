#ifndef MAHJONG_ACTION_H
#define MAHJONG_ACTION_H

#include <memory>
#include <utility>

#include "mjx.pb.h"
#include "open.h"
#include "tile.h"
#include "types.h"

namespace mjx {
class Action {
 public:
  Action() = delete;
  static bool IsValid(const mjxproto::Action& action);
  static mjxproto::Action CreateDiscard(AbsolutePos who, Tile discard);
  static std::vector<mjxproto::Action> CreateDiscards(
      AbsolutePos who, const std::vector<Tile>& discards);
  static mjxproto::Action CreateRiichi(AbsolutePos who);
  static mjxproto::Action CreateTsumo(AbsolutePos who);
  static mjxproto::Action CreateRon(AbsolutePos who);
  static mjxproto::Action CreateOpen(AbsolutePos who, Open open);
  static mjxproto::Action CreateNo(AbsolutePos who);
  static mjxproto::Action CreateNineTiles(AbsolutePos who);
  static bool Equal(const mjxproto::Action& lhs, const mjxproto::Action& rhs);
};
}  // namespace mjx

#endif  // MAHJONG_ACTION_H
