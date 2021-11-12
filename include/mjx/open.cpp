#include "mjx/open.h"

#include "mjx/internal/open.h"
#include "mjx/internal/types.h"

namespace mjx {

int Open::EventType(std::uint16_t bits) {
  auto open_type = internal::Open(bits).Type();
  if (open_type == internal::OpenType::kChi) {
    return static_cast<int>(mjxproto::EVENT_TYPE_CHI);
  }
  if (open_type == internal::OpenType::kPon) {
    return static_cast<int>(mjxproto::EVENT_TYPE_PON);
  }
  if (open_type == internal::OpenType::kKanClosed) {
    return static_cast<int>(mjxproto::EVENT_TYPE_CLOSED_KAN);
  }
  if (open_type == internal::OpenType::kKanOpened) {
    return static_cast<int>(mjxproto::EVENT_TYPE_OPEN_KAN);
  }
  if (open_type == internal::OpenType::kKanAdded) {
    return static_cast<int>(mjxproto::EVENT_TYPE_ADDED_KAN);
  }
}

int Open::From(std::uint16_t bits) {
  return static_cast<int>(internal::Open(bits).From());
};

int Open::At(std::uint16_t bits, std::size_t i) {
  return internal::Open(bits).At(i).Id();
}

std::size_t Open::Size(std::uint16_t bits) {
  return internal::Open(bits).Size();
}

std::vector<int> Open::Tiles(std::uint16_t bits) {
  std::vector<int> tiles;
  for (const internal::Tile& t : internal::Open(bits).Tiles()) {
    tiles.push_back(t.Id());
  }
  return tiles;
}

std::vector<int> Open::TilesFromHand(std::uint16_t bits) {
  std::vector<int> tiles;
  for (const internal::Tile& t : internal::Open(bits).TilesFromHand()) {
    tiles.push_back(t.Id());
  }
  return tiles;
};

int Open::StolenTile(std::uint16_t bits) {
  return internal::Open(bits).StolenTile().Id();
}

int Open::LastTile(std::uint16_t bits) {
  return internal::Open(bits).LastTile().Id();
}

std::vector<int> Open::UndiscardableTileTypes(std::uint16_t bits) {
  std::vector<int> tile_types;
  for (const internal::TileType& tt :
       internal::Open(bits).UndiscardableTileTypes()) {
    tile_types.push_back(static_cast<int>(tt));
  }
  return tile_types;
}

}  // namespace mjx