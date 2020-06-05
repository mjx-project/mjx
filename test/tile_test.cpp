#include "gtest/gtest.h"
#include "tile.h"

using namespace mj;


TEST(tile, Tile)
{
    EXPECT_NO_FATAL_FAILURE(Tile(0));
    EXPECT_NO_FATAL_FAILURE(Tile(135));
    EXPECT_EQ(Tile(tile_type::m1).Id(), 0);
    EXPECT_EQ(Tile(tile_type::m1, 3).Id(), 3);
    EXPECT_EQ(Tile(tile_type::m2).Id(), 4);
    EXPECT_EQ(Tile(tile_type::m2, 3).Id(), 7);
    EXPECT_EQ(Tile(tile_type::m3).Id(), 8);
    EXPECT_EQ(Tile(tile_type::m3, 3).Id(), 11);
    EXPECT_EQ(Tile(tile_type::m4).Id(), 12);
    EXPECT_EQ(Tile(tile_type::m4, 3).Id(), 15);
    EXPECT_EQ(Tile(tile_type::rd).Id(), 132);
    EXPECT_EQ(Tile(tile_type::rd, 3).Id(), 135);
    EXPECT_EQ(Tile("m1").Id(), 0);
    EXPECT_EQ(Tile("rd", 3).Id(), 135);
}


TEST(tile, Create)
{
    auto tiles1 = Tile::Create(std::vector<tile_id>{10, 20, 30});
    auto expected1 = std::vector<Tile>{Tile(10), Tile(20), Tile(30)};
    EXPECT_EQ(expected1, tiles1);
    auto tiles2 = Tile::Create({tile_type::m1, tile_type::m2, tile_type::m1});
    auto expected2 = std::vector<Tile>{Tile(0), Tile(4), Tile(1)};
    EXPECT_EQ(expected2, tiles2);
    auto tiles3 = Tile::Create({"m1", "m2", "m1"});
    auto expected3 = Tile::Create({tile_type::m1, tile_type::m2, tile_type::m1});
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
    EXPECT_EQ(tile_type::m1, t.Type());
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
    EXPECT_EQ(tile_set_type::manzu, m5.Color());
    auto p5 = Tile(52);
    EXPECT_EQ(tile_set_type::pinzu, p5.Color());
    auto s5 = Tile(88);
    EXPECT_EQ(tile_set_type::souzu, s5.Color());
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
    EXPECT_TRUE(t1.Is(tile_type::m1));
    EXPECT_FALSE(t1.Is(tile_type::m2));
    EXPECT_TRUE(t2.Is(tile_type::rd));

    // all
    EXPECT_TRUE(t1.Is(tile_set_type::all));
    // manzu
    auto m1 = Tile(0);
    auto m9 = Tile(35);
    EXPECT_TRUE(m1.Is(tile_type::m1));
    EXPECT_TRUE(m9.Is(tile_type::m9));
    EXPECT_TRUE(m1.Is(tile_set_type::manzu));
    // souzu
    auto s1 = Tile(72);
    auto s9 = Tile(107);
    EXPECT_TRUE(s1.Is(tile_type::s1));
    EXPECT_TRUE(s9.Is(tile_type::s9));
    EXPECT_TRUE(s1.Is(tile_set_type::souzu));
    EXPECT_TRUE(s9.Is(tile_set_type::souzu));
    EXPECT_TRUE(m9.Is(tile_set_type::manzu));
    // pinzu
    auto p1 = Tile(36);
    auto p9 = Tile(71);
    EXPECT_TRUE(p1.Is(tile_type::p1));
    EXPECT_TRUE(p9.Is(tile_type::p9));
    EXPECT_TRUE(p1.Is(tile_set_type::pinzu));
    EXPECT_TRUE(p9.Is(tile_set_type::pinzu));
    // tanyao
    auto m2 = Tile(4);
    auto m8 = Tile(31);
    auto p2 = Tile(40);
    auto p8 = Tile(67);
    auto s2 = Tile(76);
    auto s8 = Tile(103);
    EXPECT_TRUE(m2.Is(tile_type::m2));
    EXPECT_TRUE(m8.Is(tile_type::m8));
    EXPECT_TRUE(p2.Is(tile_type::p2));
    EXPECT_TRUE(p8.Is(tile_type::p8));
    EXPECT_TRUE(s2.Is(tile_type::s2));
    EXPECT_TRUE(s8.Is(tile_type::s8));
    EXPECT_TRUE(m2.Is(tile_set_type::tanyao));
    EXPECT_TRUE(m8.Is(tile_set_type::tanyao));
    EXPECT_TRUE(p2.Is(tile_set_type::tanyao));
    EXPECT_TRUE(p8.Is(tile_set_type::tanyao));
    EXPECT_TRUE(s2.Is(tile_set_type::tanyao));
    EXPECT_TRUE(s8.Is(tile_set_type::tanyao));
    // terminals
    EXPECT_TRUE(p1.Is(tile_type::p1));
    EXPECT_TRUE(p1.Is(tile_set_type::terminals));
    EXPECT_FALSE(m2.Is(tile_set_type::terminals));
    // winds
    auto ew = Tile(108);
    auto nw = Tile(123);
    EXPECT_TRUE(ew.Is(tile_type::ew));
    EXPECT_TRUE(nw.Is(tile_type::nw));
    EXPECT_TRUE(ew.Is(tile_set_type::winds));
    EXPECT_TRUE(nw.Is(tile_set_type::winds));
    // dragons
    auto wd = Tile(124);
    auto rd = Tile(135);
    EXPECT_TRUE(wd.Is(tile_type::wd));
    EXPECT_TRUE(rd.Is(tile_type::rd));
    EXPECT_TRUE(wd.Is(tile_set_type::dragons));
    EXPECT_TRUE(rd.Is(tile_set_type::dragons));
    // honors
    EXPECT_TRUE(ew.Is(tile_set_type::honors));
    EXPECT_TRUE(nw.Is(tile_set_type::honors));
    EXPECT_TRUE(wd.Is(tile_set_type::honors));
    EXPECT_TRUE(rd.Is(tile_set_type::honors));
   // yaochu
    EXPECT_TRUE(m1.Is(tile_set_type::yaochu));
    EXPECT_TRUE(m9.Is(tile_set_type::yaochu));
    EXPECT_TRUE(p1.Is(tile_set_type::yaochu));
    EXPECT_TRUE(p9.Is(tile_set_type::yaochu));
    EXPECT_TRUE(s1.Is(tile_set_type::yaochu));
    EXPECT_TRUE(s9.Is(tile_set_type::yaochu));
    EXPECT_TRUE(nw.Is(tile_set_type::yaochu));
    EXPECT_TRUE(ew.Is(tile_set_type::yaochu));
    EXPECT_TRUE(wd.Is(tile_set_type::yaochu));
    EXPECT_TRUE(rd.Is(tile_set_type::yaochu));
    // red_five
    auto m5 = Tile(16);
    auto p5 = Tile(52);
    auto s5 = Tile(88);
    EXPECT_TRUE(m5.Is(tile_type::m5));
    EXPECT_TRUE(p5.Is(tile_type::p5));
    EXPECT_TRUE(s5.Is(tile_type::s5));
    EXPECT_TRUE(m5.Is(tile_set_type::red_five));
    EXPECT_TRUE(s5.Is(tile_set_type::red_five));
    EXPECT_TRUE(p5.Is(tile_set_type::red_five));
    // empty
    EXPECT_FALSE(m1.Is(tile_set_type::empty));
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
