#include <mjx/internal/action.h>

#include "gtest/gtest.h"

using namespace mjx::internal;

TEST(internal_action, Encode) {
  // Discard
  EXPECT_EQ(
      Action::Encode(Action::CreateDiscard(AbsolutePos::kInitEast, Tile(17))),
      4);
  // Discard red
  EXPECT_EQ(
      Action::Encode(Action::CreateDiscard(AbsolutePos::kInitEast, Tile(16))),
      34);
  // Tsumogiri
  EXPECT_EQ(
      Action::Encode(Action::CreateTsumogiri(AbsolutePos::kInitEast, Tile(17))),
      41);
  // Tsumogiri red
  EXPECT_EQ(
      Action::Encode(Action::CreateTsumogiri(AbsolutePos::kInitEast, Tile(16))),
      71);
  // Chi
  auto tiles = std::vector<Tile>{Tile(12), Tile(17), Tile(20)};
  EXPECT_EQ(Action::Encode(Action::CreateOpen(AbsolutePos::kInitEast,
                                              Chi::Create(tiles, Tile(12)))),
            77);
  tiles = std::vector<Tile>{Tile(48), Tile(53), Tile(56)};
  EXPECT_EQ(Action::Encode(Action::CreateOpen(AbsolutePos::kInitEast,
                                              Chi::Create(tiles, Tile(48)))),
            84);
  // Chi w/ red
  tiles = std::vector<Tile>{Tile(12), Tile(16), Tile(20)};
  EXPECT_EQ(Action::Encode(Action::CreateOpen(AbsolutePos::kInitEast,
                                              Chi::Create(tiles, Tile(12)))),
            96);
  // Pon
  EXPECT_EQ(Action::Encode(Action::CreateOpen(
                AbsolutePos::kInitEast,
                Pon::Create(Tile(17), Tile(16), RelativePos::kLeft))),
            108);
  // Pon w/ red
  EXPECT_EQ(Action::Encode(Action::CreateOpen(
                AbsolutePos::kInitEast,
                Pon::Create(Tile(16), Tile(17), RelativePos::kLeft))),
            138);
  // Kan
  EXPECT_EQ(Action::Encode(Action::CreateOpen(AbsolutePos::kInitEast,
                                              KanClosed::Create(Tile(16)))),
            145);
  EXPECT_EQ(Action::Encode(Action::CreateOpen(
                AbsolutePos::kInitEast,
                KanOpened::Create(Tile(16), RelativePos::kLeft))),
            145);
  EXPECT_EQ(Action::Encode(Action::CreateOpen(
                AbsolutePos::kInitEast,
                KanAdded::Create(
                    Pon::Create(Tile(17), Tile(16), RelativePos::kLeft)))),
            145);
  // Tsumo
  EXPECT_EQ(Action::Encode(Action::CreateTsumo(AbsolutePos::kInitEast, Tile(3),
                                               std::string())),
            175);
  // Ron
  EXPECT_EQ(Action::Encode(Action::CreateRon(AbsolutePos::kInitEast, Tile(5),
                                             std::string())),
            176);
  // Riichi
  EXPECT_EQ(Action::Encode(Action::CreateRiichi(AbsolutePos::kInitEast)), 177);
  // Kyusyu
  EXPECT_EQ(Action::Encode(Action::CreateNineTiles(AbsolutePos::kInitEast)),
            178);
  // No
  EXPECT_EQ(Action::Encode(Action::CreateNo(AbsolutePos::kInitEast)), 179);
}
