#include <mjx/internal/action.h>

#include "gtest/gtest.h"

using namespace mjx::internal;

TEST(action, Encode) {
  // Discard
  EXPECT_EQ(
      Action::Encode(Action::CreateDiscard(AbsolutePos::kInitEast, Tile(17))),
      4);
  // Discard red
  EXPECT_EQ(
      Action::Encode(Action::CreateDiscard(AbsolutePos::kInitEast, Tile(16))),
      34);
  // Chi
  auto tiles = std::vector<Tile>{Tile(12), Tile(17), Tile(20)};
  EXPECT_EQ(Action::Encode(Action::CreateOpen(AbsolutePos::kInitEast,
                                              Chi::Create(tiles, Tile(12)))),
            40);
  tiles = std::vector<Tile>{Tile(48), Tile(53), Tile(56)};
  EXPECT_EQ(Action::Encode(Action::CreateOpen(AbsolutePos::kInitEast,
                                              Chi::Create(tiles, Tile(48)))),
            47);
  // Chi w/ red
  tiles = std::vector<Tile>{Tile(12), Tile(16), Tile(20)};
  EXPECT_EQ(Action::Encode(Action::CreateOpen(AbsolutePos::kInitEast,
                                              Chi::Create(tiles, Tile(12)))),
            59);
  // Pon
  EXPECT_EQ(Action::Encode(Action::CreateOpen(
                AbsolutePos::kInitEast,
                Pon::Create(Tile(17), Tile(16), RelativePos::kLeft))),
            71);
  // Pon w/ red
  EXPECT_EQ(Action::Encode(Action::CreateOpen(
                AbsolutePos::kInitEast,
                Pon::Create(Tile(16), Tile(17), RelativePos::kLeft))),
            101);
  // Kan
  EXPECT_EQ(Action::Encode(Action::CreateOpen(AbsolutePos::kInitEast,
                                              KanClosed::Create(Tile(16)))),
            108);
  EXPECT_EQ(Action::Encode(Action::CreateOpen(
                AbsolutePos::kInitEast,
                KanOpened::Create(Tile(16), RelativePos::kLeft))),
            108);
  EXPECT_EQ(Action::Encode(Action::CreateOpen(
                AbsolutePos::kInitEast,
                KanAdded::Create(
                    Pon::Create(Tile(17), Tile(16), RelativePos::kLeft)))),
            108);
  // Tsumo
  EXPECT_EQ(Action::Encode(Action::CreateTsumo(AbsolutePos::kInitEast)), 138);
  // Ron
  EXPECT_EQ(Action::Encode(Action::CreateRon(AbsolutePos::kInitEast)), 139);
  // Riichi
  EXPECT_EQ(Action::Encode(Action::CreateRiichi(AbsolutePos::kInitEast)), 140);
  // Kyusyu
  EXPECT_EQ(Action::Encode(Action::CreateNineTiles(AbsolutePos::kInitEast)),
            141);
  // No
  EXPECT_EQ(Action::Encode(Action::CreateNo(AbsolutePos::kInitEast)), 142);
}
