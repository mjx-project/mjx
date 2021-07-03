#include "mjx/internal/tile.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#include "mjx/internal/utils.h"

namespace mjx::internal {
Tile::Tile(TileId tile_id) : tile_id_(tile_id) { Assert(IsValid()); }

Tile::Tile(TileType tile_type, std::uint8_t offset)
    : Tile(static_cast<std::uint8_t>(tile_type) * 4 + offset) {
  Assert(offset <= 3);
}

Tile::Tile(const std::string &tile_type_str, std::uint8_t offset)
    : Tile(Str2Type(tile_type_str), offset) {}

std::vector<Tile> Tile::Create(const std::vector<TileId> &vector,
                               bool sorted) noexcept {
  auto tiles = std::vector<Tile>();
  for (const auto &id : vector) tiles.emplace_back(id);
  if (sorted) std::sort(tiles.begin(), tiles.end());
  return tiles;
}

std::vector<Tile> Tile::Create(const std::vector<TileType> &vector,
                               bool sorted) noexcept {
  std::unordered_map<TileType, std::uint8_t> m;
  auto tiles = std::vector<Tile>();
  for (const auto &type : vector) {
    if (m.find(type) == m.end()) m[type] = 0;
    TileId id = static_cast<TileId>(type) * 4 + m[type];
    tiles.emplace_back(id);
    ++m[type];
  }
  if (sorted) std::sort(tiles.begin(), tiles.end());
  return tiles;
}

std::vector<Tile> Tile::Create(const std::vector<std::string> &vector,
                               bool sorted) noexcept {
  std::vector<TileType> types;
  types.reserve(vector.size());
  for (const auto &s : vector) types.emplace_back(Tile::Str2Type(s));
  auto tiles = Tile::Create(types, sorted);
  return tiles;
}

std::vector<Tile> Tile::CreateAll() noexcept {
  // TODO: switch depending on rule::PLAYER_NUM
  auto ids = std::vector<TileId>(136);
  std::iota(ids.begin(), ids.end(), 0);
  auto tiles = Tile::Create(ids);
  return tiles;
}

TileId Tile::Id() const noexcept {
  Assert(IsValid());
  return tile_id_;
}

TileType Tile::Type() const noexcept {
  Assert(IsValid());
  return TileType(TypeUint());
}

std::uint8_t Tile::TypeUint() const noexcept {
  Assert(IsValid());
  return Id() / 4;
}

TileSetType Tile::Color() const noexcept {
  if (Is(TileSetType::kManzu)) return TileSetType::kManzu;
  if (Is(TileSetType::kPinzu)) return TileSetType::kPinzu;
  if (Is(TileSetType::kSouzu)) return TileSetType::kSouzu;
  Assert(false);
}

std::uint8_t Tile::Num() const noexcept {
  Assert(!Is(TileSetType::kHonours));
  return TypeUint() % 9 + 1;
}

bool Tile::Is(std::uint8_t n) const noexcept {
  if (Is(TileSetType::kHonours)) return false;
  return Num() == n;
}

bool Tile::Is(TileType tile_type) const noexcept { return Type() == tile_type; }

bool Tile::Is(TileSetType tile_set_type) const noexcept {
  auto tt = Type();
  switch (tile_set_type) {
    case TileSetType::kAll:
      return true;
    case TileSetType::kManzu:
      return TileType::kM1 <= tt && tt <= TileType::kM9;
    case TileSetType::kPinzu:
      return TileType::kP1 <= tt && tt <= TileType::kP9;
    case TileSetType::kSouzu:
      return TileType::kS1 <= tt && tt <= TileType::kS9;
    case TileSetType::kTanyao:
      return (TileType::kM2 <= tt && tt <= TileType::kM8) ||
             (TileType::kP2 <= tt && tt <= TileType::kP8) ||
             (TileType::kS2 <= tt && tt <= TileType::kS8);
    case TileSetType::kTerminals:
      return tt == TileType::kM1 || tt == TileType::kM9 ||
             tt == TileType::kP1 || tt == TileType::kP9 ||
             tt == TileType::kS1 || tt == TileType::kS9;
    case TileSetType::kWinds:
      return TileType::kEW <= tt && tt <= TileType::kNW;
    case TileSetType::kDragons:
      return TileType::kWD <= tt && tt <= TileType::kRD;
    case TileSetType::kHonours:
      return TileType::kEW <= tt && tt <= TileType::kRD;
    case TileSetType::kYaocyu:
      return tt == TileType::kM1 || tt == TileType::kM9 ||
             tt == TileType::kP1 || tt == TileType::kP9 ||
             tt == TileType::kS1 || tt == TileType::kS9 ||
             (TileType::kEW <= tt && tt <= TileType::kRD);
    case TileSetType::kRedFive:
      return (tt == TileType::kM5 || tt == TileType::kP5 ||
              tt == TileType::kS5) &&
             tile_id_ % 4 == 0;
    case TileSetType::kEmpty:
      return false;
  }
}

bool Tile::IsRedFive() const {
  // TODO: switch depending on rule
  return Id() == 16 || Id() == 52 || Id() == 88;
}

bool Tile::operator==(const Tile &right) const noexcept {
  return tile_id_ == right.tile_id_;
}

bool Tile::operator!=(const Tile &right) const noexcept {
  return !(*this == right);
}

bool Tile::operator<(const Tile &right) const noexcept {
  return tile_id_ < right.tile_id_;
}

bool Tile::operator<=(const Tile &right) const noexcept {
  return tile_id_ <= right.tile_id_;
}

bool Tile::operator>(const Tile &right) const noexcept {
  return tile_id_ > right.tile_id_;
}

bool Tile::operator>=(const Tile &right) const noexcept {
  return tile_id_ >= right.tile_id_;
}

std::string Tile::ToString(bool verbose) const noexcept {
  std::string s = "";
  switch (Type()) {
    case TileType::kM1:
      s += "m1";
      break;
    case TileType::kM2:
      s += "m2";
      break;
    case TileType::kM3:
      s += "m3";
      break;
    case TileType::kM4:
      s += "m4";
      break;
    case TileType::kM5:
      s += "m5";
      break;
    case TileType::kM6:
      s += "m6";
      break;
    case TileType::kM7:
      s += "m7";
      break;
    case TileType::kM8:
      s += "m8";
      break;
    case TileType::kM9:
      s += "m9";
      break;
    case TileType::kP1:
      s += "p1";
      break;
    case TileType::kP2:
      s += "p2";
      break;
    case TileType::kP3:
      s += "p3";
      break;
    case TileType::kP4:
      s += "p4";
      break;
    case TileType::kP5:
      s += "p5";
      break;
    case TileType::kP6:
      s += "p6";
      break;
    case TileType::kP7:
      s += "p7";
      break;
    case TileType::kP8:
      s += "p8";
      break;
    case TileType::kP9:
      s += "p9";
      break;
    case TileType::kS1:
      s += "s1";
      break;
    case TileType::kS2:
      s += "s2";
      break;
    case TileType::kS3:
      s += "s3";
      break;
    case TileType::kS4:
      s += "s4";
      break;
    case TileType::kS5:
      s += "s5";
      break;
    case TileType::kS6:
      s += "s6";
      break;
    case TileType::kS7:
      s += "s7";
      break;
    case TileType::kS8:
      s += "s8";
      break;
    case TileType::kS9:
      s += "s9";
      break;
    case TileType::kEW:
      s += "ew";
      break;
    case TileType::kSW:
      s += "sw";
      break;
    case TileType::kWW:
      s += "ww";
      break;
    case TileType::kNW:
      s += "nw";
      break;
    case TileType::kWD:
      s += "wd";
      break;
    case TileType::kGD:
      s += "gd";
      break;
    case TileType::kRD:
      s += "rd";
      break;
  }
  if (verbose) s += "(" + std::to_string(Offset()) + ")";
  return s;
}

std::string Tile::ToUnicode() const noexcept {
  switch (Type()) {
    case TileType::kM1:
      return u8"\U0001F007";
    case TileType::kM2:
      return u8"\U0001F008";
    case TileType::kM3:
      return u8"\U0001F009";
    case TileType::kM4:
      return u8"\U0001F00A";
    case TileType::kM5:
      return u8"\U0001F00B";
    case TileType::kM6:
      return u8"\U0001F00C";
    case TileType::kM7:
      return u8"\U0001F00D";
    case TileType::kM8:
      return u8"\U0001F00E";
    case TileType::kM9:
      return u8"\U0001F00F";
    case TileType::kP1:
      return u8"\U0001F019";
    case TileType::kP2:
      return u8"\U0001F01A";
    case TileType::kP3:
      return u8"\U0001F01B";
    case TileType::kP4:
      return u8"\U0001F01C";
    case TileType::kP5:
      return u8"\U0001F01D";
    case TileType::kP6:
      return u8"\U0001F01E";
    case TileType::kP7:
      return u8"\U0001F01F";
    case TileType::kP8:
      return u8"\U0001F020";
    case TileType::kP9:
      return u8"\U0001F021";
    case TileType::kS1:
      return u8"\U0001F010";
    case TileType::kS2:
      return u8"\U0001F011";
    case TileType::kS3:
      return u8"\U0001F012";
    case TileType::kS4:
      return u8"\U0001F013";
    case TileType::kS5:
      return u8"\U0001F014";
    case TileType::kS6:
      return u8"\U0001F015";
    case TileType::kS7:
      return u8"\U0001F016";
    case TileType::kS8:
      return u8"\U0001F017";
    case TileType::kS9:
      return u8"\U0001F018";
    case TileType::kEW:
      return u8"\U0001F000";
    case TileType::kSW:
      return u8"\U0001F001";
    case TileType::kWW:
      return u8"\U0001F002";
    case TileType::kNW:
      return u8"\U0001F003";
    case TileType::kWD:
      return u8"\U0001F006";
    case TileType::kGD:
      return u8"\U0001F005";
    case TileType::kRD:
      return u8"\U0001F004\U0000FE0E";  // Use text presentation (U+FE0E VS15)
  }
}

std::string Tile::ToChar() const noexcept {
  switch (Type()) {
    case TileType::kM1:
      return u8"一";
    case TileType::kM2:
      return u8"二";
    case TileType::kM3:
      return u8"三";
    case TileType::kM4:
      return u8"四";
    case TileType::kM5:
      return u8"五";
    case TileType::kM6:
      return u8"六";
    case TileType::kM7:
      return u8"七";
    case TileType::kM8:
      return u8"八";
    case TileType::kM9:
      return u8"九";
    case TileType::kP1:
      return u8"①";
    case TileType::kP2:
      return u8"②";
    case TileType::kP3:
      return u8"③";
    case TileType::kP4:
      return u8"④";
    case TileType::kP5:
      return u8"⑤";
    case TileType::kP6:
      return u8"⑥";
    case TileType::kP7:
      return u8"⑦";
    case TileType::kP8:
      return u8"⑧";
    case TileType::kP9:
      return u8"⑨";
    case TileType::kS1:
      return u8"１";
    case TileType::kS2:
      return u8"２";
    case TileType::kS3:
      return u8"３";
    case TileType::kS4:
      return u8"４";
    case TileType::kS5:
      return u8"５";
    case TileType::kS6:
      return u8"６";
    case TileType::kS7:
      return u8"７";
    case TileType::kS8:
      return u8"８";
    case TileType::kS9:
      return u8"９";
    case TileType::kEW:
      return u8"東";
    case TileType::kSW:
      return u8"南";
    case TileType::kWW:
      return u8"西";
    case TileType::kNW:
      return u8"北";
    case TileType::kWD:
      return u8"白";
    case TileType::kGD:
      return u8"發";
    case TileType::kRD:
      return u8"中";
  }
}

bool Tile::IsValid() const noexcept { return 0 <= tile_id_ && tile_id_ < 136; }

TileType Tile::Str2Type(const std::string &s) noexcept {
  if (s == "m1") return TileType::kM1;
  if (s == "m2") return TileType::kM2;
  if (s == "m3") return TileType::kM3;
  if (s == "m4") return TileType::kM4;
  if (s == "m5") return TileType::kM5;
  if (s == "m6") return TileType::kM6;
  if (s == "m7") return TileType::kM7;
  if (s == "m8") return TileType::kM8;
  if (s == "m9") return TileType::kM9;
  if (s == "p1") return TileType::kP1;
  if (s == "p2") return TileType::kP2;
  if (s == "p3") return TileType::kP3;
  if (s == "p4") return TileType::kP4;
  if (s == "p5") return TileType::kP5;
  if (s == "p6") return TileType::kP6;
  if (s == "p7") return TileType::kP7;
  if (s == "p8") return TileType::kP8;
  if (s == "p9") return TileType::kP9;
  if (s == "s1") return TileType::kS1;
  if (s == "s2") return TileType::kS2;
  if (s == "s3") return TileType::kS3;
  if (s == "s4") return TileType::kS4;
  if (s == "s5") return TileType::kS5;
  if (s == "s6") return TileType::kS6;
  if (s == "s7") return TileType::kS7;
  if (s == "s8") return TileType::kS8;
  if (s == "s9") return TileType::kS9;
  if (s == "ew") return TileType::kEW;
  if (s == "sw") return TileType::kSW;
  if (s == "ww") return TileType::kWW;
  if (s == "nw") return TileType::kNW;
  if (s == "wd") return TileType::kWD;
  if (s == "gd") return TileType::kGD;
  if (s == "rd") return TileType::kRD;
  Assert(false);  // TODO: fix
}

std::uint8_t Tile::Offset() const noexcept {
  return static_cast<std::uint8_t>(tile_id_) % 4;
}

bool Tile::Equals(Tile other) const noexcept {
  return Type() == other.Type() && IsRedFive() == other.IsRedFive();
}

std::string Tile::ToString(const std::vector<Tile> &tiles) noexcept {
  Assert(!tiles.empty(), "tiles should not be empty.");
  std::string s;
  for (const auto &t : tiles) {
    s += t.ToString(true) + ",";
  }
  s.pop_back();
  return s;
}
}  // namespace mjx::internal
