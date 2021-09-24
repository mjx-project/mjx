#include <mjx/internal/tile.h>

#include "gtest/gtest.h"

using namespace mjx::internal;

TEST(internal_tile, Tile) {
  EXPECT_NO_FATAL_FAILURE(Tile(0));
  EXPECT_NO_FATAL_FAILURE(Tile(135));
  EXPECT_EQ(Tile(TileType::kM1).Id(), 0);
  EXPECT_EQ(Tile(TileType::kM1, 3).Id(), 3);
  EXPECT_EQ(Tile(TileType::kM2).Id(), 4);
  EXPECT_EQ(Tile(TileType::kM2, 3).Id(), 7);
  EXPECT_EQ(Tile(TileType::kM3).Id(), 8);
  EXPECT_EQ(Tile(TileType::kM3, 3).Id(), 11);
  EXPECT_EQ(Tile(TileType::kM4).Id(), 12);
  EXPECT_EQ(Tile(TileType::kM4, 3).Id(), 15);
  EXPECT_EQ(Tile(TileType::kRD).Id(), 132);
  EXPECT_EQ(Tile(TileType::kRD, 3).Id(), 135);
  EXPECT_EQ(Tile("m1").Id(), 0);
  EXPECT_EQ(Tile("rd", 3).Id(), 135);
}

TEST(internal_tile, Create) {
  auto tiles1 = Tile::Create(std::vector<TileId>{10, 20, 30});
  auto expected1 = std::vector<Tile>{Tile(10), Tile(20), Tile(30)};
  EXPECT_EQ(expected1, tiles1);
  auto tiles2 = Tile::Create({TileType::kM1, TileType::kM2, TileType::kM1});
  auto expected2 = std::vector<Tile>{Tile(0), Tile(4), Tile(1)};
  EXPECT_EQ(expected2, tiles2);
  auto tiles3 = Tile::Create({"m1", "m2", "m1"});
  auto expected3 = Tile::Create({TileType::kM1, TileType::kM2, TileType::kM1});
  EXPECT_EQ(expected3, tiles3);
  auto tiles4 = Tile::CreateAll();
  EXPECT_EQ(Tile(0), tiles4.at(0));
  EXPECT_EQ(Tile(135), tiles4.at(135));
}

TEST(internal_tile, Id) {
  auto t = Tile(100);
  EXPECT_EQ(100, t.Id());
}

TEST(internal_tile, Type) {
  auto t = Tile(0);
  EXPECT_EQ(TileType::kM1, t.Type());
}

TEST(internal_tile, TypeUint) {
  auto t = Tile(0);
  EXPECT_EQ(t.TypeUint(), 0);
  t = Tile(135);
  EXPECT_EQ(t.TypeUint(), 33);
}

TEST(internal_tile, Offset) {
  auto t = Tile(0);
  EXPECT_EQ(t.Offset(), 0);
  t = Tile(33);
  EXPECT_EQ(t.Offset(), 1);
  t = Tile(135);
  EXPECT_EQ(t.Offset(), 3);
}

TEST(internal_tile, Color) {
  auto m5 = Tile(16);
  EXPECT_EQ(TileSetType::kManzu, m5.Color());
  auto p5 = Tile(52);
  EXPECT_EQ(TileSetType::kPinzu, p5.Color());
  auto s5 = Tile(88);
  EXPECT_EQ(TileSetType::kSouzu, s5.Color());
}

TEST(internal_tile, IsRedFive) {
  auto m5 = Tile(16);
  EXPECT_TRUE(m5.IsRedFive());
  auto p5 = Tile(52);
  EXPECT_TRUE(p5.IsRedFive());
  auto s5 = Tile(88);
  EXPECT_TRUE(s5.IsRedFive());
}

TEST(internal_tile, Num) {
  auto t = Tile("m1", 0);
  EXPECT_EQ(t.Num(), 1);
  t = Tile("m9", 3);
  EXPECT_EQ(t.Num(), 9);
  t = Tile("s5", 2);
  EXPECT_EQ(t.Num(), 5);
  t = Tile("s1", 0);
  EXPECT_EQ(t.Num(), 1);
  t = Tile("m9", 3);
  EXPECT_EQ(t.Num(), 9);
}

TEST(internal_tile, Is) {
  // num
  auto t = Tile("m1", 0);
  EXPECT_TRUE(t.Is(1));
  t = Tile("m9", 3);
  EXPECT_TRUE(t.Is(9));
  t = Tile("s5", 2);
  EXPECT_TRUE(t.Is(5));
  t = Tile("s1", 0);
  EXPECT_TRUE(t.Is(1));
  t = Tile("m9", 3);
  EXPECT_TRUE(t.Is(9));

  // type
  auto t1 = Tile(0);
  auto t2 = Tile(135);
  EXPECT_TRUE(t1.Is(TileType::kM1));
  EXPECT_FALSE(t1.Is(TileType::kM2));
  EXPECT_TRUE(t2.Is(TileType::kRD));

  // all
  EXPECT_TRUE(t1.Is(TileSetType::kAll));
  // manzu
  auto m1 = Tile(0);
  auto m9 = Tile(35);
  EXPECT_TRUE(m1.Is(TileType::kM1));
  EXPECT_TRUE(m9.Is(TileType::kM9));
  EXPECT_TRUE(m1.Is(TileSetType::kManzu));
  // souzu
  auto s1 = Tile(72);
  auto s9 = Tile(107);
  EXPECT_TRUE(s1.Is(TileType::kS1));
  EXPECT_TRUE(s9.Is(TileType::kS9));
  EXPECT_TRUE(s1.Is(TileSetType::kSouzu));
  EXPECT_TRUE(s9.Is(TileSetType::kSouzu));
  EXPECT_TRUE(m9.Is(TileSetType::kManzu));
  // pinzu
  auto p1 = Tile(36);
  auto p9 = Tile(71);
  EXPECT_TRUE(p1.Is(TileType::kP1));
  EXPECT_TRUE(p9.Is(TileType::kP9));
  EXPECT_TRUE(p1.Is(TileSetType::kPinzu));
  EXPECT_TRUE(p9.Is(TileSetType::kPinzu));
  // tanyao
  auto m2 = Tile(4);
  auto m8 = Tile(31);
  auto p2 = Tile(40);
  auto p8 = Tile(67);
  auto s2 = Tile(76);
  auto s8 = Tile(103);
  EXPECT_TRUE(m2.Is(TileType::kM2));
  EXPECT_TRUE(m8.Is(TileType::kM8));
  EXPECT_TRUE(p2.Is(TileType::kP2));
  EXPECT_TRUE(p8.Is(TileType::kP8));
  EXPECT_TRUE(s2.Is(TileType::kS2));
  EXPECT_TRUE(s8.Is(TileType::kS8));
  EXPECT_TRUE(m2.Is(TileSetType::kTanyao));
  EXPECT_TRUE(m8.Is(TileSetType::kTanyao));
  EXPECT_TRUE(p2.Is(TileSetType::kTanyao));
  EXPECT_TRUE(p8.Is(TileSetType::kTanyao));
  EXPECT_TRUE(s2.Is(TileSetType::kTanyao));
  EXPECT_TRUE(s8.Is(TileSetType::kTanyao));
  // terminals
  EXPECT_TRUE(p1.Is(TileType::kP1));
  EXPECT_TRUE(p1.Is(TileSetType::kTerminals));
  EXPECT_FALSE(m2.Is(TileSetType::kTerminals));
  // winds
  auto ew = Tile(108);
  auto nw = Tile(123);
  EXPECT_TRUE(ew.Is(TileType::kEW));
  EXPECT_TRUE(nw.Is(TileType::kNW));
  EXPECT_TRUE(ew.Is(TileSetType::kWinds));
  EXPECT_TRUE(nw.Is(TileSetType::kWinds));
  // dragons
  auto wd = Tile(124);
  auto rd = Tile(135);
  EXPECT_TRUE(wd.Is(TileType::kWD));
  EXPECT_TRUE(rd.Is(TileType::kRD));
  EXPECT_TRUE(wd.Is(TileSetType::kDragons));
  EXPECT_TRUE(rd.Is(TileSetType::kDragons));
  // honors
  EXPECT_TRUE(ew.Is(TileSetType::kHonours));
  EXPECT_TRUE(nw.Is(TileSetType::kHonours));
  EXPECT_TRUE(wd.Is(TileSetType::kHonours));
  EXPECT_TRUE(rd.Is(TileSetType::kHonours));
  // yaochu
  EXPECT_TRUE(m1.Is(TileSetType::kYaocyu));
  EXPECT_TRUE(m9.Is(TileSetType::kYaocyu));
  EXPECT_TRUE(p1.Is(TileSetType::kYaocyu));
  EXPECT_TRUE(p9.Is(TileSetType::kYaocyu));
  EXPECT_TRUE(s1.Is(TileSetType::kYaocyu));
  EXPECT_TRUE(s9.Is(TileSetType::kYaocyu));
  EXPECT_TRUE(nw.Is(TileSetType::kYaocyu));
  EXPECT_TRUE(ew.Is(TileSetType::kYaocyu));
  EXPECT_TRUE(wd.Is(TileSetType::kYaocyu));
  EXPECT_TRUE(rd.Is(TileSetType::kYaocyu));
  // red_five
  auto m5 = Tile(16);
  auto p5 = Tile(52);
  auto s5 = Tile(88);
  EXPECT_TRUE(m5.Is(TileType::kM5));
  EXPECT_TRUE(p5.Is(TileType::kP5));
  EXPECT_TRUE(s5.Is(TileType::kS5));
  EXPECT_TRUE(m5.Is(TileSetType::kRedFive));
  EXPECT_TRUE(s5.Is(TileSetType::kRedFive));
  EXPECT_TRUE(p5.Is(TileSetType::kRedFive));
  // empty
  EXPECT_FALSE(m1.Is(TileSetType::kEmpty));
}

TEST(internal_tile, ComparisonOperators) {
  auto t1 = Tile(0);
  auto t2 = Tile(0);
  auto t3 = Tile(50);
  // ==
  EXPECT_TRUE(t1 == t2);
  EXPECT_FALSE(t1 == t3);
  // !=
  EXPECT_TRUE(t1 != t3);
  EXPECT_FALSE(t1 != t2);
  // <
  EXPECT_TRUE(t1 < t3);
  EXPECT_FALSE(t1 < t2);
  // <=
  EXPECT_TRUE(t1 <= t3);
  EXPECT_TRUE(t1 <= t2);
  EXPECT_FALSE(t3 <= t1);
  // >
  EXPECT_TRUE(t3 > t1);
  EXPECT_FALSE(t1 > t2);
  // >=
  EXPECT_TRUE(t3 >= t1);
  EXPECT_TRUE(t1 >= t2);
  EXPECT_FALSE(t1 >= t3);
}

TEST(internal_tile, ToString) {
  auto t1 = Tile(0);
  auto t2 = Tile(135);
  EXPECT_EQ("m1", t1.ToString());
  EXPECT_EQ("rd", t2.ToString());
  EXPECT_EQ("m1(0)", t1.ToString(true));
  EXPECT_EQ("rd(3)", t2.ToString(true));
}

TEST(internal_tile, Equals) {
  auto t1 = Tile("m5", 0);
  auto t2 = Tile("m5", 1);
  auto t3 = Tile("m5", 2);
  EXPECT_FALSE(t1.Equals(t2));
  EXPECT_TRUE(t2.Equals(t3));
  t1 = Tile("m6", 0);
  t2 = Tile("m6", 1);
  EXPECT_TRUE(t1.Equals(t2));
  t1 = Tile("p5", 0);
  t2 = Tile("p5", 1);
  t3 = Tile("p5", 2);
  EXPECT_FALSE(t1.Equals(t2));
  EXPECT_TRUE(t2.Equals(t3));
  t1 = Tile("s5", 0);
  t2 = Tile("s5", 1);
  t3 = Tile("s5", 2);
  EXPECT_FALSE(t1.Equals(t2));
  EXPECT_TRUE(t2.Equals(t3));
  t1 = Tile("s5", 0);
  t2 = Tile("s5", 0);
  EXPECT_TRUE(t1.Equals(t2));
}
