#ifndef MAHJONG_ACTION_H
#define MAHJONG_ACTION_H

#include <memory>
#include <optional>
#include <utility>

#include "mjx/internal/mjx.pb.h"
#include "mjx/internal/open.h"
#include "mjx/internal/tile.h"
#include "mjx/internal/types.h"

namespace mjx::internal {
class Action {
 public:
  Action() = delete;
  static bool IsValid(const mjxproto::Action& action);
  static mjxproto::Action CreateDiscard(AbsolutePos who, Tile discard,
                                        std::string game_id = "");
  static mjxproto::Action CreateTsumogiri(AbsolutePos who, Tile discard,
                                          std::string game_id = "");
  static std::vector<mjxproto::Action> CreateDiscardsAndTsumogiri(
      AbsolutePos who, const std::vector<std::pair<Tile, bool>>& discards,
      std::string game_id = "");
  static mjxproto::Action CreateRiichi(AbsolutePos who,
                                       std::string game_id = "");
  static mjxproto::Action CreateTsumo(AbsolutePos who, Tile tile,
                                      std::string game_id = "");
  static mjxproto::Action CreateRon(AbsolutePos who, Tile tile,
                                    std::string game_id = "");
  static mjxproto::Action CreateOpen(AbsolutePos who, Open open,
                                     std::string game_id = "");
  static mjxproto::Action CreateNo(AbsolutePos who, std::string game_id = "");
  static mjxproto::Action CreateNineTiles(AbsolutePos who,
                                          std::string game_id = "");
  static mjxproto::Action CreateDummy(AbsolutePos who,
                                      std::string game_id = "");
  static std::optional<mjxproto::Action> FromEvent(
      const mjxproto::Event& event);
  static std::string ProtoToJson(const mjxproto::Action& proto);
  static bool Equal(const mjxproto::Action& lhs, const mjxproto::Action& rhs);
  static std::uint8_t Encode(const mjxproto::Action& action);
  static mjxproto::Action Decode(
      std::uint8_t code, const std::vector<mjxproto::Action>& legal_action);
  // 0~33: Discard m1~rd
  // 34,35,36: Discard m5(red), p5(red), s5(red)
  // 37~70: Tsumogiri m1~rd
  // 71,72,73: Tsumogiri m5(red), p5(red), s5(red)
  // 74~94: Chi m1m2m3 ~ s7s8s9
  // 95,96,97: Chi m3m4m5(red), m4m5(red)m6, m5(red)m6m7
  // 98,99,100: Chi p3p4p5(red), p4p5(red)p6, p5(red)p6p7
  // 101,102,103: Chi s3s4s5(red), s4s5(red)s6, s5(red)s6s7
  // 104~137: Pon m1~rd
  // 138,139,140: Pon m5(w/ red), s5(w/ red), p5(w/ red)
  // 141~174: Kan m1~rd
  // 175: Tsumo
  // 176: Ron
  // 177: Riichi
  // 178: Kyuushu
  // 179: No
  // 180: Dummy
};
}  // namespace mjx::internal

#endif  // MAHJONG_ACTION_H
