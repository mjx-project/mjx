#include "mjx/internal/open.h"

#include <algorithm>
#include <bitset>
#include <cassert>

#include "mjx/internal/utils.h"

namespace mjx::internal {
// References
//   - https://github.com/NegativeMjark/tenhou-log
//   - http://m77.hatenablog.com/entry/2017/05/21/214529
constexpr std::uint16_t MASK_FROM = 0b0000000000000011;
constexpr std::uint16_t MASK_IS_CHI = 0b0000000000000100;           // >>2
constexpr std::uint16_t MASK_CHI_OFFSET[3] = {0b0000000000011000,   // >>3
                                              0b0000000001100000,   // >>5
                                              0b0000000110000000};  // >>7
constexpr std::uint16_t MASK_CHI_BASE_AND_STOLEN =
    0b1111110000000000;  // >>10, ((type/9)*7+type%9)*3+(stolen)
constexpr std::uint16_t MASK_IS_PON = 0b0000000000001000;             // >>3
constexpr std::uint16_t MASK_IS_KAN_ADDED = 0b0000000000010000;       // >>4
constexpr std::uint16_t MASK_PON_UNUSED_OFFSET = 0b0000000001100000;  // >>5
constexpr std::uint16_t MASK_PON_BASE_AND_STOLEN =
    0b1111111000000000;  // >>9, type*3+(stolen)
constexpr std::uint16_t MASK_KAN_STOLEN = 0b1111111100000000;  // >>8, id

Open::Open(std::uint16_t bits) : bits_(bits) {}

OpenType Open::Type() const {
  if (bits_ & MASK_IS_CHI) {
    return OpenType::kChi;
  } else if (bits_ & MASK_IS_PON) {
    return OpenType::kPon;
  } else if (bits_ & MASK_IS_KAN_ADDED) {
    return OpenType::kKanAdded;
  } else {
    if (RelativePos(static_cast<std::uint8_t>(bits_ & MASK_FROM)) ==
        RelativePos::kSelf) {
      return OpenType::kKanClosed;
    } else {
      return OpenType::kKanOpened;
    }
  }
}

std::uint16_t Open::GetBits() const { return bits_; }

std::string Open::ToString(bool verbose) const {
  std::string s = "[";
  for (const auto &t : Tiles()) {
    s += t.ToString(verbose) + ",";
  }
  s.pop_back();
  s += "]";
  if (Type() == OpenType::kKanOpened) s += "o";
  if (Type() == OpenType::kKanClosed) s += "c";
  if (Type() == OpenType::kKanAdded) s += "a";
  return s;
}

RelativePos Open::From() const {
  switch (Type()) {
    case OpenType::kChi:
      return Chi::From(bits_);
    case OpenType::kPon:
      return Pon::From(bits_);
    case OpenType::kKanOpened:
      return KanOpened::From(bits_);
    case OpenType::kKanClosed:
      return KanClosed::From(bits_);
    case OpenType::kKanAdded:
      return KanAdded::From(bits_);
  }
}
Tile Open::At(std::size_t i) const {
  switch (Type()) {
    case OpenType::kChi:
      return Chi::At(bits_, i);
    case OpenType::kPon:
      return Pon::At(bits_, i);
    case OpenType::kKanOpened:
      return KanOpened::At(bits_, i);
    case OpenType::kKanClosed:
      return KanClosed::At(bits_, i);
    case OpenType::kKanAdded:
      return KanAdded::At(bits_, i);
  }
}
std::size_t Open::Size() const {
  switch (Type()) {
    case OpenType::kChi:
      return Chi::Size(bits_);
    case OpenType::kPon:
      return Pon::Size(bits_);
    case OpenType::kKanOpened:
      return KanOpened::Size(bits_);
    case OpenType::kKanClosed:
      return KanClosed::Size(bits_);
    case OpenType::kKanAdded:
      return KanAdded::Size(bits_);
  }
}
std::vector<Tile> Open::Tiles() const {
  switch (Type()) {
    case OpenType::kChi:
      return Chi::Tiles(bits_);
    case OpenType::kPon:
      return Pon::Tiles(bits_);
    case OpenType::kKanOpened:
      return KanOpened::Tiles(bits_);
    case OpenType::kKanClosed:
      return KanClosed::Tiles(bits_);
    case OpenType::kKanAdded:
      return KanAdded::Tiles(bits_);
  }
}
std::vector<Tile> Open::TilesFromHand() const {
  switch (Type()) {
    case OpenType::kChi:
      return Chi::TilesFromHand(bits_);
    case OpenType::kPon:
      return Pon::TilesFromHand(bits_);
    case OpenType::kKanOpened:
      return KanOpened::TilesFromHand(bits_);
    case OpenType::kKanClosed:
      return KanClosed::TilesFromHand(bits_);
    case OpenType::kKanAdded:
      return KanAdded::TilesFromHand(bits_);
  }
}
Tile Open::StolenTile() const {
  switch (Type()) {
    case OpenType::kChi:
      return Chi::StolenTile(bits_);
    case OpenType::kPon:
      return Pon::StolenTile(bits_);
    case OpenType::kKanOpened:
      return KanOpened::StolenTile(bits_);
    case OpenType::kKanClosed:
      return KanClosed::StolenTile(bits_);
    case OpenType::kKanAdded:
      return KanAdded::StolenTile(bits_);
  }
}
Tile Open::LastTile() const {
  switch (Type()) {
    case OpenType::kChi:
      return Chi::LastTile(bits_);
    case OpenType::kPon:
      return Pon::LastTile(bits_);
    case OpenType::kKanOpened:
      return KanOpened::LastTile(bits_);
    case OpenType::kKanClosed:
      return KanClosed::LastTile(bits_);
    case OpenType::kKanAdded:
      return KanAdded::LastTile(bits_);
  }
}
std::vector<TileType> Open::UndiscardableTileTypes() const {
  switch (Type()) {
    case OpenType::kChi:
      return Chi::UndiscardableTileTypes(bits_);
    case OpenType::kPon:
      return Pon::UndiscardableTileTypes(bits_);
    case OpenType::kKanOpened:
      return KanOpened::UndiscardableTileTypes(bits_);
    case OpenType::kKanClosed:
      return KanClosed::UndiscardableTileTypes(bits_);
    case OpenType::kKanAdded:
      return KanAdded::UndiscardableTileTypes(bits_);
  }
}

bool Open::operator==(Open other) const noexcept {
  return bits_ == other.bits_;
}

bool Open::operator!=(Open other) const noexcept { return !(*this == other); }

bool Open::Equals(Open other) const noexcept {
  if (Type() != other.Type() || From() != other.From()) return false;
  switch (Type()) {
    case OpenType::kChi:
      return At(0).Equals(other.At(0)) && At(1).Equals(other.At(1)) &&
             At(2).Equals(other.At(2));
    case OpenType::kPon:
    case OpenType::kKanOpened:
      return StolenTile().Equals(other.StolenTile());
    case OpenType::kKanClosed:
      return At(0).Equals(other.At(0));
    case OpenType::kKanAdded:
      return StolenTile().Equals(other.StolenTile()) &&
             LastTile().Equals(other.LastTile());
  }
}

// Chi

Open Chi::Create(std::vector<Tile> &tiles, Tile stolen) {
  std::sort(tiles.begin(), tiles.end());
  std::uint16_t bits = 0;
  bits |= (MASK_FROM & static_cast<std::uint16_t>(RelativePos::kLeft));
  bits |= MASK_IS_CHI;
  bits |= (static_cast<std::uint16_t>(tiles.at(0).Id() % 4) << 3);
  bits |= (static_cast<std::uint16_t>(tiles.at(1).Id() % 4) << 5);
  bits |= (static_cast<std::uint16_t>(tiles.at(2).Id() % 4) << 7);
  std::uint16_t base = tiles.at(0).Id() / 4;
  std::uint16_t stolen_ix = std::distance(
      tiles.begin(), std::find(tiles.begin(), tiles.end(), stolen));
  bits |=
      static_cast<std::uint16_t>(((base / 9) * 7 + base % 9) * 3 + stolen_ix)
      << 10;
  return Open(bits);
}

RelativePos Chi::From(std::uint16_t bits) { return RelativePos::kLeft; }
Tile Chi::At(std::uint16_t bits, std::size_t i) {
  Assert(i < 3);
  return at(bits, i, min_type(bits));
}
std::size_t Chi::Size(std::uint16_t bits) { return 3; }
std::vector<Tile> Chi::Tiles(std::uint16_t bits) {
  auto v = std::vector<Tile>();
  auto m = min_type(bits);
  for (std::size_t i = 0; i < 3; ++i) v.push_back(at(bits, i, m));
  return v;
}
std::vector<Tile> Chi::TilesFromHand(std::uint16_t bits) {
  auto v = std::vector<Tile>();
  auto m = min_type(bits);
  for (std::size_t i = 0; i < 3; ++i) {
    if (i == (bits >> 10) % 3) continue;
    v.push_back(at(bits, i, m));
  }
  return v;
}
Tile Chi::StolenTile(std::uint16_t bits) { return At(bits, (bits >> 10) % 3); }
Tile Chi::LastTile(std::uint16_t bits) { return StolenTile(bits); }
std::vector<TileType> Chi::UndiscardableTileTypes(std::uint16_t bits) {
  auto v = std::vector<TileType>();
  auto stolen_ = StolenTile(bits);
  auto type = stolen_.Type();
  v.push_back(type);
  // m2m3[m4]
  if (At(bits, 2) == stolen_ &&
      ((TileType::kM4 <= type && type <= TileType::kM9) ||
       (TileType::kP4 <= type && type <= TileType::kP9) ||
       (TileType::kS4 <= type && type <= TileType::kS9))) {
    auto prev = TileType(static_cast<std::uint8_t>(type) - 3);
    v.push_back(prev);
  }
  // [m6]m7m8
  if (At(bits, 0) == stolen_ &&
      ((TileType::kM1 <= type && type <= TileType::kM6) ||
       (TileType::kP1 <= type && type <= TileType::kP6) ||
       (TileType::kS1 <= type && type <= TileType::kS6))) {
    auto next = TileType(static_cast<std::uint8_t>(type) + 3);
    v.push_back(next);
  }
  return v;
}

std::uint16_t Chi::min_type(std::uint16_t bits) {
  std::uint16_t min_type_base21 = (bits >> 10) / 3;
  return (min_type_base21 / 7) * 9 + min_type_base21 % 7;
}

Tile Chi::at(std::uint16_t bits, std::size_t i, std::uint16_t min_type) {
  return Tile(
      static_cast<std::uint8_t>((min_type + static_cast<std::uint8_t>(i)) * 4 +
                                ((bits & MASK_CHI_OFFSET[i]) >> (2 * i + 3))));
}

// end Chi

// Pon

Open Pon::Create(Tile stolen, Tile unused, RelativePos from) {
  Assert(stolen.Type() == unused.Type());
  std::uint16_t bits = 0;
  bits |= (MASK_FROM & static_cast<std::uint16_t>(from));
  bits |= MASK_IS_PON;
  std::uint16_t unused_offset = static_cast<std::uint16_t>(unused.Id() % 4);
  bits |= unused_offset << 5;
  std::uint16_t base = static_cast<std::uint16_t>(stolen.Type());
  // stolen\unused
  //     0  1  2  3
  // 0   -  0  0  0
  // 1   0  -  1  1
  // 2   1  1  -  2
  // 3   2  2  2  -
  std::uint16_t stolen_ix = static_cast<std::uint16_t>(stolen.Id() % 4);
  if (stolen_ix > unused_offset) --stolen_ix;
  Assert(stolen_ix < 3);
  bits |= (base * 3 + stolen_ix) << 9;
  return Open(bits);
}

RelativePos Pon::From(std::uint16_t bits) {
  return RelativePos(static_cast<std::uint8_t>(bits & MASK_FROM));
}
Tile Pon::At(std::uint16_t bits, std::size_t i) {
  std::uint16_t type = (bits >> 9) / 3;
  std::uint16_t unused_offset = (bits & MASK_PON_UNUSED_OFFSET) >> 5;
  if (i >= unused_offset) ++i;
  // unused at(0) at(1) at(2)
  // 0 [1]  2   3
  // 1  0  [2]  3
  // 2  0   1  [3]
  // 3  0   1   2
  return Tile(static_cast<std::uint8_t>(type * 4 + i));
}
std::size_t Pon::Size(std::uint16_t bits) { return 3; }
std::vector<Tile> Pon::Tiles(std::uint16_t bits) {
  auto v = std::vector<Tile>();
  for (std::size_t i = 0; i < 3; ++i) v.push_back(At(bits, i));
  return v;
}
std::vector<Tile> Pon::TilesFromHand(std::uint16_t bits) {
  auto v = std::vector<Tile>();
  std::uint16_t stolen_ix = (bits >> 9) % 3;
  for (std::size_t i = 0; i < 3; ++i)
    if (i != stolen_ix) v.push_back(At(bits, i));
  return v;
}
Tile Pon::StolenTile(std::uint16_t bits) {
  std::uint16_t stolen_ix = (bits >> 9) % 3;
  return At(bits, stolen_ix);
}
Tile Pon::LastTile(std::uint16_t bits) { return StolenTile(bits); }
std::vector<TileType> Pon::UndiscardableTileTypes(std::uint16_t bits) {
  return std::vector<TileType>(1, At(bits, 0).Type());
}

// end Pon

// KanOpened

Open KanOpened::Create(Tile stolen, RelativePos from) {
  std::uint16_t bits = 0;
  bits |= static_cast<std::uint16_t>(from);
  bits |= (static_cast<std::uint16_t>(stolen.Id()) << 8);
  return Open(bits);
}

RelativePos KanOpened::From(std::uint16_t bits) {
  return RelativePos(static_cast<std::uint8_t>(bits & MASK_FROM));
}
Tile KanOpened::At(std::uint16_t bits, std::size_t i) {
  return Tile(static_cast<std::uint8_t>(((bits >> 8) / 4) * 4 + i));
}
std::size_t KanOpened::Size(std::uint16_t bits) { return 4; }
std::vector<Tile> KanOpened::Tiles(std::uint16_t bits) {
  auto v = std::vector<TileType>(
      4, TileType(static_cast<std::uint8_t>((bits >> 8) / 4)));
  return Tile::Create(v, true);
}
std::vector<Tile> KanOpened::TilesFromHand(std::uint16_t bits) {
  auto v = std::vector<Tile>();
  auto type = (bits >> 8) / 4;
  auto stolen_offset = (bits >> 8) % 4;
  for (std::size_t i = 0; i < 4; ++i)
    if (i != stolen_offset)
      v.push_back(Tile(static_cast<std::uint8_t>(type * 4 + i)));
  return v;
}
Tile KanOpened::StolenTile(std::uint16_t bits) {
  return Tile(static_cast<std::uint8_t>(bits >> 8));
}
Tile KanOpened::LastTile(std::uint16_t bits) { return StolenTile(bits); }
std::vector<TileType> KanOpened::UndiscardableTileTypes(std::uint16_t bits) {
  return std::vector<TileType>();
}

// end KanOpened

// KanClosed

Open KanClosed::Create(Tile tile) {
  std::uint16_t bits = 0;
  bits |= static_cast<std::uint16_t>(RelativePos::kSelf);
  bits |= (static_cast<std::uint16_t>(tile.Id()) << 8);
  return Open(bits);
}

RelativePos KanClosed::From(std::uint16_t bits) { return RelativePos::kSelf; }
Tile KanClosed::At(std::uint16_t bits, std::size_t i) {
  return Tile(static_cast<std::uint8_t>(((bits >> 8) / 4) * 4 + i));
}
std::size_t KanClosed::Size(std::uint16_t bits) { return 4; }
std::vector<Tile> KanClosed::Tiles(std::uint16_t bits) {
  auto v = std::vector<TileType>(
      4, TileType(static_cast<std::uint8_t>((bits >> 8) / 4)));
  return Tile::Create(v, true);
}
std::vector<Tile> KanClosed::TilesFromHand(std::uint16_t bits) {
  return Tiles(bits);
}
Tile KanClosed::StolenTile(std::uint16_t bits) {
  return Tile(static_cast<std::uint8_t>(bits >> 8));
}
Tile KanClosed::LastTile(std::uint16_t bits) { return StolenTile(bits); }
std::vector<TileType> KanClosed::UndiscardableTileTypes(std::uint16_t bits) {
  return std::vector<TileType>();
}

// end KanClosed

// KanAdded

Open KanAdded::Create(Open pon) {
  std::uint16_t bits = pon.GetBits();
  bits |= MASK_IS_KAN_ADDED;
  bits &= ~MASK_IS_PON;
  return Open(bits);
}

RelativePos KanAdded::From(std::uint16_t bits) {
  return RelativePos(static_cast<std::uint8_t>(bits & MASK_FROM));
}
Tile KanAdded::At(std::uint16_t bits, std::size_t i) {
  Assert(i < 4);
  std::uint16_t type = (bits >> 9) / 3;
  return Tile(static_cast<std::uint8_t>(type * 4 + i));
}
std::size_t KanAdded::Size(std::uint16_t bits) { return 4; }
std::vector<Tile> KanAdded::Tiles(std::uint16_t bits) {
  std::vector<TileType> v(4,
                          TileType(static_cast<std::uint8_t>((bits >> 9) / 3)));
  return Tile::Create(v, true);
}
std::vector<Tile> KanAdded::TilesFromHand(std::uint16_t bits) {
  auto v = std::vector<Tile>();
  auto stolen = StolenTile(bits);
  for (int i = 0; i < 4; ++i) {
    auto t = At(bits, i);
    if (t != stolen) v.push_back(At(bits, i));
  }
  return v;
}
Tile KanAdded::StolenTile(std::uint16_t bits) {
  std::uint16_t type = (bits >> 9) / 3;
  std::uint16_t stolen_ix = (bits >> 9) % 3;
  std::uint16_t unused_offset = (bits & MASK_PON_UNUSED_OFFSET) >> 5;
  if (stolen_ix >= unused_offset) ++stolen_ix;
  return Tile(static_cast<std::uint8_t>(type * 4 + stolen_ix));
}
Tile KanAdded::LastTile(std::uint16_t bits) {
  std::uint16_t type = (bits >> 9) / 3;
  std::uint16_t unused_offset = (bits & MASK_PON_UNUSED_OFFSET) >> 5;
  return Tile(static_cast<std::uint8_t>(type * 4 + unused_offset));
}
std::vector<TileType> KanAdded::UndiscardableTileTypes(std::uint16_t bits) {
  return std::vector<TileType>();
}

// end KanAdded
}  // namespace mjx::internal
