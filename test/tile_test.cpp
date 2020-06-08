#include "gtest/gtest.h"
#include "tile.h"

using namespace mj;


TEST(tile, Tile)
{
    EXPECT_NO_FATAL_FAILURE(Tile(0));
    EXPECT_NO_FATAL_FAILURE(Tile(135));
    EXPECT_EQ(Tile(TileType::m1).Id(), 0);
    EXPECT_EQ(Tile(TileType::m1, 3).Id(), 3);
    EXPECT_EQ(Tile(TileType::m2).Id(), 4);
    EXPECT_EQ(Tile(TileType::m2, 3).Id(), 7);
    EXPECT_EQ(Tile(TileType::m3).Id(), 8);
    EXPECT_EQ(Tile(TileType::m3, 3).Id(), 11);
    EXPECT_EQ(Tile(TileType::m4).Id(), 12);
    EXPECT_EQ(Tile(TileType::m4, 3).Id(), 15);
    EXPECT_EQ(Tile(TileType::rd).Id(), 132);
    EXPECT_EQ(Tile(TileType::rd, 3).Id(), 135);
    EXPECT_EQ(Tile("m1").Id(), 0);
    EXPECT_EQ(Tile("rd", 3).Id(), 135);
}


TEST(tile, Create)
{
    auto tiles1 = Tile::Create(std::vector<TileId>{10, 20, 30});
    auto expected1 = std::vector<Tile>{Tile(10), Tile(20), Tile(30)};
    EXPECT_EQ(expected1, tiles1);
    auto tiles2 = Tile::Create({TileType::m1, TileType::m2, TileType::m1});
    auto expected2 = std::vector<Tile>{Tile(0), Tile(4), Tile(1)};
    EXPECT_EQ(expected2, tiles2);
    auto tiles3 = Tile::Create({"m1", "m2", "m1"});
    auto expected3 = Tile::Create({TileType::m1, TileType::m2, TileType::m1});
    EXPECT_EQ(expected3, tiles3);
    auto tiles4 = Tile::CreateAll();
    EXPECT_EQ(Tile(0), tiles4.at(0));
    EXPECT_EQ(Tile(135), tiles4.at(135));
}


TEST(tile, Id)
{
    auto t = Tile(100);
    EXPECT_EQ(100, t.Id());
}


TEST(tile, Type)
{
    auto t = Tile(0);
    EXPECT_EQ(TileType::m1, t.Type());
}

TEST(tile, TypeUint)
{
    auto t = Tile(0);
    EXPECT_EQ(t.TypeUint(), 0);
    t = Tile(135);
    EXPECT_EQ(t.TypeUint(), 33);
}

TEST(tile, Color)
{
    auto m5 = Tile(16);
    EXPECT_EQ(TileSetType::manzu, m5.Color());
    auto p5 = Tile(52);
    EXPECT_EQ(TileSetType::pinzu, p5.Color());
    auto s5 = Tile(88);
    EXPECT_EQ(TileSetType::souzu, s5.Color());
}


TEST(tile, IsRedFive)
{
    auto m5 = Tile(16);
    EXPECT_TRUE(m5.IsRedFive());
    auto p5 = Tile(52);
    EXPECT_TRUE(p5.IsRedFive());
    auto s5 = Tile(88);
    EXPECT_TRUE(s5.IsRedFive());
}

TEST(tile, Num) {
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

TEST(tile, Is)
{
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
    EXPECT_TRUE(t1.Is(TileType::m1));
    EXPECT_FALSE(t1.Is(TileType::m2));
    EXPECT_TRUE(t2.Is(TileType::rd));

    // all
    EXPECT_TRUE(t1.Is(TileSetType::all));
    // manzu
    auto m1 = Tile(0);
    auto m9 = Tile(35);
    EXPECT_TRUE(m1.Is(TileType::m1));
    EXPECT_TRUE(m9.Is(TileType::m9));
    EXPECT_TRUE(m1.Is(TileSetType::manzu));
    // souzu
    auto s1 = Tile(72);
    auto s9 = Tile(107);
    EXPECT_TRUE(s1.Is(TileType::s1));
    EXPECT_TRUE(s9.Is(TileType::s9));
    EXPECT_TRUE(s1.Is(TileSetType::souzu));
    EXPECT_TRUE(s9.Is(TileSetType::souzu));
    EXPECT_TRUE(m9.Is(TileSetType::manzu));
    // pinzu
    auto p1 = Tile(36);
    auto p9 = Tile(71);
    EXPECT_TRUE(p1.Is(TileType::p1));
    EXPECT_TRUE(p9.Is(TileType::p9));
    EXPECT_TRUE(p1.Is(TileSetType::pinzu));
    EXPECT_TRUE(p9.Is(TileSetType::pinzu));
    // tanyao
    auto m2 = Tile(4);
    auto m8 = Tile(31);
    auto p2 = Tile(40);
    auto p8 = Tile(67);
    auto s2 = Tile(76);
    auto s8 = Tile(103);
    EXPECT_TRUE(m2.Is(TileType::m2));
    EXPECT_TRUE(m8.Is(TileType::m8));
    EXPECT_TRUE(p2.Is(TileType::p2));
    EXPECT_TRUE(p8.Is(TileType::p8));
    EXPECT_TRUE(s2.Is(TileType::s2));
    EXPECT_TRUE(s8.Is(TileType::s8));
    EXPECT_TRUE(m2.Is(TileSetType::tanyao));
    EXPECT_TRUE(m8.Is(TileSetType::tanyao));
    EXPECT_TRUE(p2.Is(TileSetType::tanyao));
    EXPECT_TRUE(p8.Is(TileSetType::tanyao));
    EXPECT_TRUE(s2.Is(TileSetType::tanyao));
    EXPECT_TRUE(s8.Is(TileSetType::tanyao));
    // terminals
    EXPECT_TRUE(p1.Is(TileType::p1));
    EXPECT_TRUE(p1.Is(TileSetType::terminals));
    EXPECT_FALSE(m2.Is(TileSetType::terminals));
    // winds
    auto ew = Tile(108);
    auto nw = Tile(123);
    EXPECT_TRUE(ew.Is(TileType::ew));
    EXPECT_TRUE(nw.Is(TileType::nw));
    EXPECT_TRUE(ew.Is(TileSetType::winds));
    EXPECT_TRUE(nw.Is(TileSetType::winds));
    // dragons
    auto wd = Tile(124);
    auto rd = Tile(135);
    EXPECT_TRUE(wd.Is(TileType::wd));
    EXPECT_TRUE(rd.Is(TileType::rd));
    EXPECT_TRUE(wd.Is(TileSetType::dragons));
    EXPECT_TRUE(rd.Is(TileSetType::dragons));
    // honors
    EXPECT_TRUE(ew.Is(TileSetType::honors));
    EXPECT_TRUE(nw.Is(TileSetType::honors));
    EXPECT_TRUE(wd.Is(TileSetType::honors));
    EXPECT_TRUE(rd.Is(TileSetType::honors));
   // yaochu
    EXPECT_TRUE(m1.Is(TileSetType::yaochu));
    EXPECT_TRUE(m9.Is(TileSetType::yaochu));
    EXPECT_TRUE(p1.Is(TileSetType::yaochu));
    EXPECT_TRUE(p9.Is(TileSetType::yaochu));
    EXPECT_TRUE(s1.Is(TileSetType::yaochu));
    EXPECT_TRUE(s9.Is(TileSetType::yaochu));
    EXPECT_TRUE(nw.Is(TileSetType::yaochu));
    EXPECT_TRUE(ew.Is(TileSetType::yaochu));
    EXPECT_TRUE(wd.Is(TileSetType::yaochu));
    EXPECT_TRUE(rd.Is(TileSetType::yaochu));
    // red_five
    auto m5 = Tile(16);
    auto p5 = Tile(52);
    auto s5 = Tile(88);
    EXPECT_TRUE(m5.Is(TileType::m5));
    EXPECT_TRUE(p5.Is(TileType::p5));
    EXPECT_TRUE(s5.Is(TileType::s5));
    EXPECT_TRUE(m5.Is(TileSetType::red_five));
    EXPECT_TRUE(s5.Is(TileSetType::red_five));
    EXPECT_TRUE(p5.Is(TileSetType::red_five));
    // empty
    EXPECT_FALSE(m1.Is(TileSetType::empty));
}


TEST(tile, ComparisonOperators)
{
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


TEST(tile, ToString)
{
    auto t1 = Tile(0);
    auto t2 = Tile(135);
    EXPECT_EQ("<tile_id: 0, tile_type: 0>", t1.ToString());
    EXPECT_EQ("<tile_id: 135, tile_type: 33>", t2.ToString());
}
