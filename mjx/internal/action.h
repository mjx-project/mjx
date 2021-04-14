#ifndef MAHJONG_ACTION_H
#define MAHJONG_ACTION_H

#include <memory>
#include <utility>

#include "mjx.pb.h"
#include "open.h"
#include "tile.h"
#include "types.h"

namespace mjx::internal {
class Action {
 public:
  Action() = delete;
  static bool IsValid(const mjxproto::Action& action);
  static mjxproto::Action CreateDiscard(AbsolutePos who, Tile discard);
  static mjxproto::Action CreateTsumogiri(AbsolutePos who, Tile discard);
  static std::vector<mjxproto::Action> CreateDiscardsAndTsumogiri(
      AbsolutePos who, const std::vector<std::pair<Tile, bool>>& discards);
  static mjxproto::Action CreateRiichi(AbsolutePos who);
  static mjxproto::Action CreateTsumo(AbsolutePos who);
  static mjxproto::Action CreateRon(AbsolutePos who);
  static mjxproto::Action CreateOpen(AbsolutePos who, Open open);
  static mjxproto::Action CreateNo(AbsolutePos who);
  static mjxproto::Action CreateNineTiles(AbsolutePos who);
  static bool Equal(const mjxproto::Action& lhs, const mjxproto::Action& rhs);
  static std::uint8_t Encode(const mjxproto::Action& action);
  static mjxproto::Action Decode(
      std::uint8_t code, const std::vector<mjxproto::Action>& possible_action);
  // 0~33: Discard m1~rd
  // 34,35,36: Discard m5(red), p5(red), s5(red)
  // 37~57: Chi m1m2m3 ~ s7s8s9
  // 58,59,60: Chi m3m4m5(red), m4m5(red)m6, m5(red)m6m7
  // 61,62,63: Chi p3p4p5(red), p4p5(red)p6, p5(red)p6p7
  // 64,65,66: Chi s3s4s5(red), s4s5(red)s6, s5(red)s6s7
  // 67~100: Pon m1~rd
  // 101,102,103: Pon m5(w/ red), s5(w/ red), p5(w/ red)
  // 104~137: Kan m1~rd
  // 138: Tsumo
  // 139: Ron
  // 140: Riichi
  // 141: Kyuushu
  // 142: No
};
}  // namespace mjx::internal

#endif  // MAHJONG_ACTION_H
