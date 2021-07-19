#include <mjx/internal/open.h>

#include "gtest/gtest.h"

using namespace mjx::internal;

using tt = TileType;
auto vec_type_eq = [](std::vector<Tile> v1, std::vector<tt> v2) {
  for (int i = 0; i < v1.size(); ++i)
    if (v1.at(i).Type() != v2.at(i)) return false;
  return true;
};

TEST(internal_open, Chi) {
  // constructor
  std::vector<Tile> t = {Tile("p5", 2), Tile("p6", 1), Tile("p7", 0)};
  auto c = Chi::Create(t, Tile("p6", 1));
  EXPECT_EQ(c.Type(), OpenType::kChi);
  EXPECT_EQ(c.From(), RelativePos::kLeft);
  EXPECT_EQ(c.At(0).Id(), Tile("p5", 2).Id());
  EXPECT_EQ(c.At(1).Id(), Tile("p6", 1).Id());
  EXPECT_EQ(c.At(2).Id(), Tile("p7", 0).Id());
  EXPECT_EQ(c.StolenTile().Id(), Tile("p6", 1).Id());

  // samples from Tenhou  TODO: add more test cases
  c = Open(static_cast<std::uint16_t>(
      49495));  // http://tenhou.net/5/?log=2019101219gm-0009-0000-a1515896
  EXPECT_EQ(c.Type(), OpenType::kChi);
  EXPECT_EQ(c.From(), RelativePos::kLeft);
  EXPECT_EQ(c.At(0).Type(), TileType::kS3);
  EXPECT_EQ(c.At(1).Type(), TileType::kS4);
  EXPECT_EQ(c.At(2).Type(), TileType::kS5);
  EXPECT_EQ(c.StolenTile().Type(), TileType::kS3);
  EXPECT_EQ(c.LastTile().Type(), TileType::kS3);
  EXPECT_EQ(c.Tiles().size(), 3);
  EXPECT_EQ(c.Tiles().at(0).Type(), TileType::kS3);
  EXPECT_EQ(c.Tiles().at(1).Type(), TileType::kS4);
  EXPECT_EQ(c.Tiles().at(2).Type(), TileType::kS5);
  EXPECT_EQ(c.TilesFromHand().at(0).Type(), TileType::kS4);
  EXPECT_EQ(c.TilesFromHand().at(1).Type(), TileType::kS5);
  EXPECT_TRUE(vec_type_eq((c.Tiles()), {tt::kS3, tt::kS4, tt::kS5}));
  EXPECT_EQ(c.TilesFromHand().size(), 2);
  EXPECT_TRUE(vec_type_eq(c.TilesFromHand(), {tt::kS4, tt::kS5}));
  EXPECT_EQ(c.UndiscardableTileTypes().size(), 2);
  EXPECT_EQ(c.UndiscardableTileTypes().at(0), TileType::kS3);
  EXPECT_EQ(c.UndiscardableTileTypes().at(1), TileType::kS6);
  EXPECT_EQ(c.ToString(), "[s3,s4,s5]");
  EXPECT_EQ(c.ToString(true), "[s3(2),s4(2),s5(2)]");

  // undiscardable_tile_types
  t = Tile::Create({"m4", "m3", "m2"});
  c = Chi::Create(t, Tile("m4"));
  EXPECT_EQ(c.UndiscardableTileTypes().size(), 2);
  EXPECT_EQ(c.UndiscardableTileTypes().at(0), TileType::kM4);
  EXPECT_EQ(c.UndiscardableTileTypes().at(1), TileType::kM1);

  t = Tile::Create({"p7", "p8", "p6"});
  c = Chi::Create(t, Tile("p6"));
  EXPECT_EQ(c.UndiscardableTileTypes().size(), 2);
  EXPECT_EQ(c.UndiscardableTileTypes().at(0), TileType::kP6);
  EXPECT_EQ(c.UndiscardableTileTypes().at(1), TileType::kP9);

  t = Tile::Create({"s4", "s6", "s5"});
  c = Chi::Create(t, Tile("s5"));
  EXPECT_EQ(c.UndiscardableTileTypes().size(), 1);
  EXPECT_EQ(c.UndiscardableTileTypes().at(0), TileType::kS5);
}

TEST(internal_open, Pon) {
  // constructor
  auto p = Pon::Create(Tile("gd", 2), Tile("gd", 1), RelativePos::kMid);
  EXPECT_EQ(p.Type(), OpenType::kPon);
  EXPECT_EQ(p.From(), RelativePos::kMid);
  EXPECT_EQ(p.At(0).Id(), Tile("gd", 0).Id());
  EXPECT_EQ(p.At(1).Id(), Tile("gd", 2).Id());
  EXPECT_EQ(p.At(2).Id(), Tile("gd", 3).Id());
  EXPECT_EQ(p.StolenTile().Id(), Tile("gd", 2).Id());
  EXPECT_EQ(p.Tiles().size(), 3);
  EXPECT_EQ(p.Tiles().at(0).Id(), Tile("gd", 0).Id());
  EXPECT_EQ(p.Tiles().at(1).Id(), Tile("gd", 2).Id());
  EXPECT_EQ(p.Tiles().at(2).Id(), Tile("gd", 3).Id());
  EXPECT_EQ(p.TilesFromHand().at(0).Id(), Tile("gd", 0).Id());
  EXPECT_EQ(p.TilesFromHand().at(1).Id(), Tile("gd", 3).Id());
  EXPECT_TRUE(vec_type_eq(p.Tiles(), {tt::kGD, tt::kGD, tt::kGD}));
  EXPECT_EQ(p.TilesFromHand().size(), 2);
  EXPECT_TRUE(vec_type_eq(p.TilesFromHand(), {tt::kGD, tt::kGD}));
  EXPECT_EQ(p.LastTile().Id(), Tile("gd", 2).Id());
  EXPECT_EQ(p.UndiscardableTileTypes().size(), 1);
  EXPECT_EQ(p.UndiscardableTileTypes().at(0), TileType::kGD);

  // samples from Tenhou  TODO: add more test cases
  p = Open(static_cast<std::uint16_t>(
      47723));  // http://tenhou.net/5/?log=2019101219gm-0009-0000-a1515896
  EXPECT_EQ(p.Type(), OpenType::kPon);
  EXPECT_EQ(p.From(), RelativePos::kLeft);
  EXPECT_EQ(p.At(0).Type(), TileType::kWD);
  EXPECT_EQ(p.At(1).Type(), TileType::kWD);
  EXPECT_EQ(p.At(2).Type(), TileType::kWD);
  EXPECT_EQ(p.StolenTile().Type(), TileType::kWD);
  EXPECT_EQ(p.LastTile().Type(), TileType::kWD);
  EXPECT_EQ(p.UndiscardableTileTypes().size(), 1);
  EXPECT_EQ(p.UndiscardableTileTypes().at(0), TileType::kWD);
  EXPECT_EQ(p.ToString(), "[wd,wd,wd]");
  EXPECT_EQ(p.ToString(true), "[wd(0),wd(1),wd(2)]");
}

TEST(internal_open, KanOpened) {
  auto k = KanOpened::Create(Tile("m2", 3), RelativePos::kRight);
  EXPECT_EQ(k.Type(), OpenType::kKanOpened);
  EXPECT_EQ(k.From(), RelativePos::kRight);
  EXPECT_EQ(k.At(0).Id(), Tile("m2", 0).Id());
  EXPECT_EQ(k.At(1).Id(), Tile("m2", 1).Id());
  EXPECT_EQ(k.At(2).Id(), Tile("m2", 2).Id());
  EXPECT_EQ(k.At(3).Id(), Tile("m2", 3).Id());
  EXPECT_EQ(k.Tiles().size(), 4);
  EXPECT_TRUE(vec_type_eq(k.Tiles(), {tt::kM2, tt::kM2, tt::kM2, tt::kM2}));
  EXPECT_EQ(k.Tiles().at(0).Id(), Tile("m2", 0).Id());
  EXPECT_EQ(k.Tiles().at(1).Id(), Tile("m2", 1).Id());
  EXPECT_EQ(k.Tiles().at(2).Id(), Tile("m2", 2).Id());
  EXPECT_EQ(k.Tiles().at(3).Id(), Tile("m2", 3).Id());
  EXPECT_EQ(k.TilesFromHand().size(), 3);
  EXPECT_EQ(k.TilesFromHand().at(0).Id(), Tile("m2", 0).Id());
  EXPECT_EQ(k.TilesFromHand().at(1).Id(), Tile("m2", 1).Id());
  EXPECT_EQ(k.TilesFromHand().at(2).Id(), Tile("m2", 2).Id());
  EXPECT_TRUE(vec_type_eq(k.TilesFromHand(), {tt::kM2, tt::kM2, tt::kM2}));
  EXPECT_EQ(k.StolenTile().Type(), TileType::kM2);
  EXPECT_EQ(k.LastTile().Type(), TileType::kM2);
  EXPECT_EQ(k.UndiscardableTileTypes().size(), 0);
  EXPECT_EQ(k.ToString(), "[m2,m2,m2,m2]o");
  EXPECT_EQ(k.ToString(true), "[m2(0),m2(1),m2(2),m2(3)]o");
}

TEST(internal_open, KanClosed) {
  auto k = KanClosed::Create(Tile("m3"));
  EXPECT_EQ(k.Type(), OpenType::kKanClosed);
  EXPECT_EQ(k.From(), RelativePos::kSelf);
  EXPECT_EQ(k.At(0).Id(), Tile("m3", 0).Id());
  EXPECT_EQ(k.At(1).Id(), Tile("m3", 1).Id());
  EXPECT_EQ(k.At(2).Id(), Tile("m3", 2).Id());
  EXPECT_EQ(k.At(3).Id(), Tile("m3", 3).Id());
  EXPECT_EQ(k.Tiles().size(), 4);
  EXPECT_TRUE(vec_type_eq(k.Tiles(), {tt::kM3, tt::kM3, tt::kM3, tt::kM3}));
  EXPECT_EQ(k.Tiles().at(0).Id(), Tile("m3", 0).Id());
  EXPECT_EQ(k.Tiles().at(1).Id(), Tile("m3", 1).Id());
  EXPECT_EQ(k.Tiles().at(2).Id(), Tile("m3", 2).Id());
  EXPECT_EQ(k.Tiles().at(3).Id(), Tile("m3", 3).Id());
  EXPECT_EQ(k.TilesFromHand().size(), 4);
  EXPECT_EQ(k.TilesFromHand().at(0).Id(), Tile("m3", 0).Id());
  EXPECT_EQ(k.TilesFromHand().at(1).Id(), Tile("m3", 1).Id());
  EXPECT_EQ(k.TilesFromHand().at(2).Id(), Tile("m3", 2).Id());
  EXPECT_EQ(k.TilesFromHand().at(3).Id(), Tile("m3", 3).Id());
  EXPECT_TRUE(vec_type_eq(k.Tiles(), {tt::kM3, tt::kM3, tt::kM3, tt::kM3}));
  EXPECT_EQ(k.StolenTile().Type(), TileType::kM3);
  EXPECT_EQ(k.LastTile().Type(), TileType::kM3);
  EXPECT_EQ(k.UndiscardableTileTypes().size(), 0);
  EXPECT_EQ(k.ToString(), "[m3,m3,m3,m3]c");
  EXPECT_EQ(k.ToString(true), "[m3(0),m3(1),m3(2),m3(3)]c");
}

TEST(internal_open, KanAdded) {
  auto p = Pon::Create(Tile("m1", 2), Tile("m1", 0), RelativePos::kMid);
  auto k = KanAdded::Create(p);
  EXPECT_EQ(k.Type(), OpenType::kKanAdded);
  EXPECT_EQ(k.From(), RelativePos::kMid);
  EXPECT_EQ(k.At(0).Id(), Tile("m1", 0).Id());
  EXPECT_EQ(k.At(1).Id(), Tile("m1", 1).Id());
  EXPECT_EQ(k.At(2).Id(), Tile("m1", 2).Id());
  EXPECT_EQ(k.At(3).Id(), Tile("m1", 3).Id());
  EXPECT_EQ(k.StolenTile().Type(), TileType::kM1);
  EXPECT_EQ(k.StolenTile().Id(), Tile("m1", 2).Id());
  EXPECT_EQ(k.Tiles().size(), 4);
  EXPECT_TRUE(vec_type_eq(p.Tiles(), {tt::kM1, tt::kM1, tt::kM1, tt::kM1}));
  EXPECT_EQ(k.Tiles().at(0).Id(), Tile("m1", 0).Id());
  EXPECT_EQ(k.Tiles().at(1).Id(), Tile("m1", 1).Id());
  EXPECT_EQ(k.Tiles().at(2).Id(), Tile("m1", 2).Id());
  EXPECT_EQ(k.Tiles().at(3).Id(), Tile("m1", 3).Id());
  EXPECT_EQ(k.TilesFromHand().size(), 3);
  EXPECT_TRUE(vec_type_eq(p.Tiles(), {tt::kM1, tt::kM1, tt::kM1}));
  EXPECT_EQ(k.TilesFromHand().at(0).Id(), Tile("m1", 0).Id());
  EXPECT_EQ(k.TilesFromHand().at(1).Id(), Tile("m1", 1).Id());
  EXPECT_EQ(k.TilesFromHand().at(2).Id(), Tile("m1", 3).Id());
  EXPECT_EQ(k.LastTile().Type(), TileType::kM1);
  EXPECT_EQ(k.LastTile().Id(), Tile("m1", 0).Id());
  EXPECT_EQ(k.UndiscardableTileTypes().size(), 0);
  EXPECT_EQ(k.ToString(), "[m1,m1,m1,m1]a");
  EXPECT_EQ(k.ToString(true), "[m1(0),m1(1),m1(2),m1(3)]a");
}

TEST(internal_open, OpenGenerator) {
  auto check = [](Open o, std::vector<tt> v, tt t, RelativePos f) {
    auto tiles = o.Tiles();
    for (int i = 0; i < v.size(); ++i)
      if (o.At(i).Type() != v.at(i)) return false;
    if (o.StolenTile().Type() != t) return false;
    if (o.From() != f) return false;
    return true;
  };
  // http://tenhou.net/5/?log=2019101219gm-0009-0000-a1515896
  // EAST 1 0
  auto o = Open(47723);
  EXPECT_EQ(o.Type(), OpenType::kPon);
  EXPECT_TRUE(
      check(o, {tt::kWD, tt::kWD, tt::kWD}, tt::kWD, RelativePos::kLeft));
  o = Open(51306);
  EXPECT_EQ(o.Type(), OpenType::kPon);
  EXPECT_TRUE(
      check(o, {tt::kRD, tt::kRD, tt::kRD}, tt::kRD, RelativePos::kMid));
  o = Open(49495);
  EXPECT_EQ(o.Type(), OpenType::kChi);
  EXPECT_TRUE(
      check(o, {tt::kS3, tt::kS4, tt::kS5}, tt::kS3, RelativePos::kLeft));
  o = Open(3146);
  EXPECT_EQ(o.Type(), OpenType::kPon);
  EXPECT_TRUE(
      check(o, {tt::kM3, tt::kM3, tt::kM3}, tt::kM3, RelativePos::kMid));
  // EAST 1 1
  o = Open(42058);
  EXPECT_EQ(o.Type(), OpenType::kPon);
  EXPECT_TRUE(
      check(o, {tt::kEW, tt::kEW, tt::kEW}, tt::kEW, RelativePos::kMid));
  o = Open(40489);
  EXPECT_EQ(o.Type(), OpenType::kPon);
  EXPECT_TRUE(
      check(o, {tt::kS9, tt::kS9, tt::kS9}, tt::kS9, RelativePos::kRight));
  o = Open(34911);
  EXPECT_EQ(o.Type(), OpenType::kChi);
  EXPECT_TRUE(
      check(o, {tt::kP5, tt::kP6, tt::kP7}, tt::kP6, RelativePos::kLeft));
  o = Open(27178);
  EXPECT_EQ(o.Type(), OpenType::kPon);
  EXPECT_TRUE(
      check(o, {tt::kP9, tt::kP9, tt::kP9}, tt::kP9, RelativePos::kMid));
  o = Open(37063);
  EXPECT_EQ(o.Type(), OpenType::kChi);
  EXPECT_TRUE(
      check(o, {tt::kP6, tt::kP7, tt::kP8}, tt::kP6, RelativePos::kLeft));
  // East 1 2
  o = Open(12905);
  EXPECT_EQ(o.Type(), OpenType::kPon);
  EXPECT_TRUE(
      check(o, {tt::kM9, tt::kM9, tt::kM9}, tt::kM9, RelativePos::kRight));
  o = Open(51753);
  EXPECT_EQ(o.Type(), OpenType::kPon);
  EXPECT_TRUE(
      check(o, {tt::kRD, tt::kRD, tt::kRD}, tt::kRD, RelativePos::kRight));
  o = Open(50679);
  EXPECT_EQ(o.Type(), OpenType::kChi);
  EXPECT_TRUE(
      check(o, {tt::kS3, tt::kS4, tt::kS5}, tt::kS4, RelativePos::kLeft));
  o = Open(14679);
  EXPECT_EQ(o.Type(), OpenType::kChi);
  EXPECT_TRUE(
      check(o, {tt::kM5, tt::kM6, tt::kM7}, tt::kM7, RelativePos::kLeft));
  // East 1 3
  o = Open(43);
  EXPECT_EQ(o.Type(), OpenType::kPon);
  EXPECT_TRUE(
      check(o, {tt::kM1, tt::kM1, tt::kM1}, tt::kM1, RelativePos::kLeft));
  o = Open(7583);
  EXPECT_EQ(o.Type(), OpenType::kChi);
  EXPECT_TRUE(
      check(o, {tt::kM3, tt::kM4, tt::kM5}, tt::kM4, RelativePos::kLeft));
  // East 1 4
  o = Open(4649);
  EXPECT_EQ(o.Type(), OpenType::kPon);
  EXPECT_TRUE(
      check(o, {tt::kM4, tt::kM4, tt::kM4}, tt::kM4, RelativePos::kRight));
  o = Open(2063);
  EXPECT_EQ(o.Type(), OpenType::kChi);
  EXPECT_TRUE(
      check(o, {tt::kM1, tt::kM2, tt::kM3}, tt::kM3, RelativePos::kLeft));
  o = Open(20615);
  EXPECT_EQ(o.Type(), OpenType::kChi);
  EXPECT_TRUE(
      check(o, {tt::kM7, tt::kM8, tt::kM9}, tt::kM9, RelativePos::kLeft));
  // East 2 0
  o = Open(42539);
  EXPECT_EQ(o.Type(), OpenType::kPon);
  EXPECT_TRUE(
      check(o, {tt::kEW, tt::kEW, tt::kEW}, tt::kEW, RelativePos::kLeft));
  // East 2 1
  o = Open(46633);
  EXPECT_EQ(o.Type(), OpenType::kPon);
  EXPECT_TRUE(
      check(o, {tt::kNW, tt::kNW, tt::kNW}, tt::kNW, RelativePos::kRight));
  o = Open(41481);
  EXPECT_EQ(o.Type(), OpenType::kPon);
  EXPECT_TRUE(
      check(o, {tt::kEW, tt::kEW, tt::kEW}, tt::kEW, RelativePos::kRight));
  o = Open(31241);
  EXPECT_EQ(o.Type(), OpenType::kPon);
  EXPECT_TRUE(
      check(o, {tt::kS3, tt::kS3, tt::kS3}, tt::kS3, RelativePos::kRight));
  // East 2 2
  // East 2 3
  o = Open(47690);
  EXPECT_EQ(o.Type(), OpenType::kPon);
  EXPECT_TRUE(
      check(o, {tt::kWD, tt::kWD, tt::kWD}, tt::kWD, RelativePos::kMid));
  // East 2 4
  o = Open(48683);
  EXPECT_EQ(o.Type(), OpenType::kPon);
  EXPECT_TRUE(
      check(o, {tt::kWD, tt::kWD, tt::kWD}, tt::kWD, RelativePos::kLeft));
  o = Open(39399);
  EXPECT_EQ(o.Type(), OpenType::kChi);
  EXPECT_TRUE(
      check(o, {tt::kP6, tt::kP7, tt::kP8}, tt::kP8, RelativePos::kLeft));
  o = Open(52303);
  EXPECT_EQ(o.Type(), OpenType::kChi);
  EXPECT_TRUE(
      check(o, {tt::kS4, tt::kS5, tt::kS6}, tt::kS4, RelativePos::kLeft));
  // EAST 3 0
  o = Open(43081);
  EXPECT_EQ(o.Type(), OpenType::kPon);
  EXPECT_TRUE(
      check(o, {tt::kSW, tt::kSW, tt::kSW}, tt::kSW, RelativePos::kRight));
  o = Open(42058);
  EXPECT_EQ(o.Type(), OpenType::kPon);
  EXPECT_TRUE(
      check(o, {tt::kEW, tt::kEW, tt::kEW}, tt::kEW, RelativePos::kMid));
  // EAST 4 0
  o = Open(60751);
  EXPECT_EQ(o.Type(), OpenType::kChi);
  EXPECT_TRUE(
      check(o, {tt::kS6, tt::kS7, tt::kS8}, tt::kS8, RelativePos::kLeft));
  o = Open(47625);
  EXPECT_EQ(o.Type(), OpenType::kPon);
  EXPECT_TRUE(
      check(o, {tt::kWD, tt::kWD, tt::kWD}, tt::kWD, RelativePos::kRight));
  // SOUTH 1 0
  o = Open(26187);
  EXPECT_EQ(o.Type(), OpenType::kPon);
  EXPECT_TRUE(
      check(o, {tt::kP9, tt::kP9, tt::kP9}, tt::kP9, RelativePos::kLeft));
  o = Open(49770);
  EXPECT_EQ(o.Type(), OpenType::kPon);
  EXPECT_TRUE(
      check(o, {tt::kGD, tt::kGD, tt::kGD}, tt::kGD, RelativePos::kMid));
  // SOUTH 2 0
  o = Open(36459);
  EXPECT_EQ(o.Type(), OpenType::kPon);
  EXPECT_TRUE(
      check(o, {tt::kS6, tt::kS6, tt::kS6}, tt::kS6, RelativePos::kLeft));

  // KanAdded from 東３局
  // https://tenhou.net/0/?log=2010112714gm-00a9-0000-d497e395
  o = Open(28722);
  EXPECT_EQ(o.Type(), OpenType::kKanAdded);
  // TODO: add tests from tenhou log for kans
}

TEST(internal_open, Equals) {
  // Chi
  std::vector<Tile> t1 = {Tile("p5", 2), Tile("p6", 1), Tile("p7", 0)};
  auto o1 = Chi::Create(t1, Tile("p6", 1));
  std::vector<Tile> t2 = {Tile("p5", 3), Tile("p6", 1), Tile("p7", 2)};
  auto o2 = Chi::Create(t2, Tile("p6", 1));
  EXPECT_TRUE(o1 != o2);
  EXPECT_TRUE(o1.Equals(o2));
  // Pon
  o1 = Pon::Create(Tile("gd", 2), Tile("gd", 0), RelativePos::kMid);
  o2 = Pon::Create(Tile("gd", 3), Tile("gd", 1), RelativePos::kMid);
  EXPECT_TRUE(o1 != o2);
  EXPECT_TRUE(o1.Equals(o2));
  o1 = Pon::Create(Tile("gd", 2), Tile("gd", 0), RelativePos::kMid);
  o2 = Pon::Create(Tile("gd", 3), Tile("gd", 1), RelativePos::kLeft);
  EXPECT_TRUE(o1 != o2);
  EXPECT_FALSE(o1.Equals(o2));
  o1 = Pon::Create(Tile("m5", 0), Tile("m5", 1), RelativePos::kMid);  // 0, 2, 3
  o2 = Pon::Create(Tile("m5", 1), Tile("m5", 2), RelativePos::kMid);  // 0, 1, 2
  EXPECT_TRUE(o1 != o2);
  EXPECT_FALSE(o1.Equals(o2));
  o1 = Pon::Create(Tile("m5", 2), Tile("m5", 0), RelativePos::kMid);
  o2 = Pon::Create(Tile("m5", 3), Tile("m5", 0), RelativePos::kLeft);
  EXPECT_TRUE(o1 != o2);
  EXPECT_FALSE(o1.Equals(o2));
  // KanClosed
  o1 = KanClosed::Create(Tile("p5", 0));
  o2 = KanClosed::Create(Tile("p5", 1));
  EXPECT_TRUE(o1 != o2);
  EXPECT_TRUE(o1.Equals(o2));
  // KanOpened
  o1 = KanOpened::Create(Tile("s5", 2), RelativePos::kRight);
  o2 = KanOpened::Create(Tile("s5", 3), RelativePos::kRight);
  EXPECT_TRUE(o1 != o2);
  EXPECT_TRUE(o1.Equals(o2));
  o1 = KanOpened::Create(Tile("s5", 0), RelativePos::kRight);
  o2 = KanOpened::Create(Tile("s5", 3), RelativePos::kRight);
  EXPECT_TRUE(o1 != o2);
  EXPECT_FALSE(o1.Equals(o2));
  // KanAdded
  o1 = Pon::Create(Tile("m5", 2), Tile("m5", 3), RelativePos::kMid);
  o1 = KanAdded::Create(o1);
  o2 = Pon::Create(Tile("m5", 3), Tile("m5", 2), RelativePos::kMid);
  o2 = KanAdded::Create(o2);
  EXPECT_TRUE(o1 != o2);
  EXPECT_TRUE(o1.Equals(o2));
  o1 = Pon::Create(Tile("m5", 2), Tile("m5", 3), RelativePos::kMid);
  o1 = KanAdded::Create(o1);
  o2 = Pon::Create(Tile("m5", 0), Tile("m5", 2), RelativePos::kMid);
  o2 = KanAdded::Create(o2);
  EXPECT_TRUE(o1 != o2);
  EXPECT_FALSE(o1.Equals(o2));
}
