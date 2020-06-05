#include "gtest/gtest.h"
#include "open.h"

using namespace mj;


using tt = tile_type;
auto vec_type_eq = [] (std::vector<Tile> v1, std::vector<tt> v2) {
    for (int i = 0; i < v1.size(); ++i)
        if (v1.at(i).Type() != v2.at(i)) return false;
    return true;
};


TEST(open, Chi)
{
    // constructor
    std::vector<Tile> t = {Tile("p5", 2), Tile("p6", 1), Tile("p7", 0)};
    std::unique_ptr<Open> c = std::make_unique<Chi>(t, Tile("p6", 1));
    EXPECT_EQ(c->type(), open_type::chi);
    EXPECT_EQ(c->from(), relative_pos::left);
    EXPECT_EQ(c->at(0).Id(), Tile("p5", 2).Id());
    EXPECT_EQ(c->at(1).Id(), Tile("p6", 1).Id());
    EXPECT_EQ(c->at(2).Id(), Tile("p7", 0).Id());
    EXPECT_EQ(c->stolen().Id(), Tile("p6", 1).Id());

    // samples from Tenhou  TODO: add more test cases
    c = std::make_unique<Chi>(static_cast<std::uint16_t>(49495));  // http://tenhou.net/5/?log=2019101219gm-0009-0000-a1515896
    EXPECT_EQ(c->type(), open_type::chi);
    EXPECT_EQ(c->from(), relative_pos::left);
    EXPECT_EQ(c->at(0).Type(), tile_type::s3);
    EXPECT_EQ(c->at(1).Type(), tile_type::s4);
    EXPECT_EQ(c->at(2).Type(), tile_type::s5);
    EXPECT_EQ(c->stolen().Type(), tile_type::s3);
    EXPECT_EQ(c->last().Type(), tile_type::s3);
    EXPECT_EQ(c->tiles().size(), 3);
    EXPECT_TRUE(vec_type_eq((c->tiles()), {tt::s3, tt::s4, tt::s5}));
    EXPECT_EQ(c->tiles_from_hand().size(), 2);
    EXPECT_TRUE(vec_type_eq(c->tiles_from_hand(), {tt::s4, tt::s5}));
    EXPECT_EQ(c->undiscardable_tile_types().size(), 2);
    EXPECT_EQ(c->undiscardable_tile_types().at(0), tile_type::s3);
    EXPECT_EQ(c->undiscardable_tile_types().at(1), tile_type::s6);

    // undiscardable_tile_types
    t = Tile::Create({"m4", "m3", "m2"});
    c = std::make_unique<Chi>(t, Tile("m4"));
    EXPECT_EQ(c->undiscardable_tile_types().size(), 2);
    EXPECT_EQ(c->undiscardable_tile_types().at(0), tile_type::m4);
    EXPECT_EQ(c->undiscardable_tile_types().at(1), tile_type::m1);

    t = Tile::Create({"p7", "p8", "p6"});
    c = std::make_unique<Chi>(t, Tile("p6"));
    EXPECT_EQ(c->undiscardable_tile_types().size(), 2);
    EXPECT_EQ(c->undiscardable_tile_types().at(0), tile_type::p6);
    EXPECT_EQ(c->undiscardable_tile_types().at(1), tile_type::p9);

    t = Tile::Create({"s4", "s6", "s5"});
    c = std::make_unique<Chi>(t, Tile("s5"));
    EXPECT_EQ(c->undiscardable_tile_types().size(), 1);
    EXPECT_EQ(c->undiscardable_tile_types().at(0), tile_type::s5);
}

TEST(open, Pon)
{
    // constructor
    std::unique_ptr<Open> p = std::make_unique<Pon>(Tile("gd", 2), Tile("gd", 1), relative_pos::mid);
    EXPECT_EQ(p->type(), open_type::pon);
    EXPECT_EQ(p->from(), relative_pos::mid);
    EXPECT_EQ(p->at(0).Id(), Tile("gd", 0).Id());
    EXPECT_EQ(p->at(1).Id(), Tile("gd", 2).Id());
    EXPECT_EQ(p->at(2).Id(), Tile("gd", 3).Id());
    EXPECT_EQ(p->stolen().Id(), Tile("gd", 2).Id());
    EXPECT_EQ(p->tiles().size(), 3);
    EXPECT_TRUE(vec_type_eq(p->tiles(), {tt::gd, tt::gd, tt::gd}));
    EXPECT_EQ(p->tiles_from_hand().size(), 2);
    EXPECT_TRUE(vec_type_eq(p->tiles_from_hand(), {tt::gd, tt::gd}));
    EXPECT_EQ(p->last().Id(), Tile("gd", 2).Id());
    EXPECT_EQ(p->undiscardable_tile_types().size(), 1);
    EXPECT_EQ(p->undiscardable_tile_types().at(0), tile_type::gd);

    // samples from Tenhou  TODO: add more test cases
    p = std::make_unique<Pon>(static_cast<std::uint16_t>(47723));  // http://tenhou.net/5/?log=2019101219gm-0009-0000-a1515896
    EXPECT_EQ(p->type(), open_type::pon);
    EXPECT_EQ(p->from(), relative_pos::left);
    EXPECT_EQ(p->at(0).Type(), tile_type::wd);
    EXPECT_EQ(p->at(1).Type(), tile_type::wd);
    EXPECT_EQ(p->at(2).Type(), tile_type::wd);
    EXPECT_EQ(p->stolen().Type(), tile_type::wd);
    EXPECT_EQ(p->last().Type(), tile_type::wd);
    EXPECT_EQ(p->undiscardable_tile_types().size(), 1);
    EXPECT_EQ(p->undiscardable_tile_types().at(0), tile_type::wd);
}

TEST(open, KanOpened)
{
    std::unique_ptr<Open> k = std::make_unique<KanOpened>(Tile("m2", 3), relative_pos::right);
    EXPECT_EQ(k->type(), open_type::kan_opened);
    EXPECT_EQ(k->from(), relative_pos::right);
    EXPECT_EQ(k->at(0).Id(), Tile("m2", 0).Id());
    EXPECT_EQ(k->at(1).Id(), Tile("m2", 1).Id());
    EXPECT_EQ(k->at(2).Id(), Tile("m2", 2).Id());
    EXPECT_EQ(k->at(3).Id(), Tile("m2", 3).Id());
    EXPECT_EQ(k->tiles().size(), 4);
    EXPECT_TRUE(vec_type_eq(k->tiles(), {tt::m2, tt::m2, tt::m2, tt::m2}));
    EXPECT_EQ(k->tiles_from_hand().size(), 3);
    EXPECT_TRUE(vec_type_eq(k->tiles_from_hand(), {tt::m2, tt::m2, tt::m2}));
    EXPECT_EQ(k->stolen().Type(), tile_type::m2);
    EXPECT_EQ(k->last().Type(), tile_type::m2);
    EXPECT_EQ(k->undiscardable_tile_types().size(), 0);
}

TEST(open, KanClosed)
{
    std::unique_ptr<Open> k = std::make_unique<KanClosed>(Tile("m3"));
    EXPECT_EQ(k->type(), open_type::kan_closed);
    EXPECT_EQ(k->from(), relative_pos::self);
    EXPECT_EQ(k->at(0).Id(), Tile("m3", 0).Id());
    EXPECT_EQ(k->at(1).Id(), Tile("m3", 1).Id());
    EXPECT_EQ(k->at(2).Id(), Tile("m3", 2).Id());
    EXPECT_EQ(k->at(3).Id(), Tile("m3", 3).Id());
    EXPECT_EQ(k->tiles().size(), 4);
    EXPECT_TRUE(vec_type_eq(k->tiles(), {tt::m3, tt::m3, tt::m3, tt::m3}));
    EXPECT_EQ(k->tiles_from_hand().size(), 4);
    EXPECT_TRUE(vec_type_eq(k->tiles(), {tt::m3, tt::m3, tt::m3, tt::m3}));
    EXPECT_EQ(k->stolen().Type(), tile_type::m3);
    EXPECT_EQ(k->last().Type(), tile_type::m3);
    EXPECT_EQ(k->undiscardable_tile_types().size(), 0);
}

TEST(open, KanAdded)
{
    std::unique_ptr<Open> p = std::make_unique<Pon>(Tile("m1", 2), Tile("m1", 0), relative_pos::mid);
    std::unique_ptr<Open> k = std::make_unique<KanAdded>(p.get());
    EXPECT_EQ(k->type(), open_type::kan_added);
    EXPECT_EQ(k->from(), relative_pos::mid);
    EXPECT_EQ(k->at(0).Id(), Tile("m1", 0).Id());
    EXPECT_EQ(k->at(1).Id(), Tile("m1", 1).Id());
    EXPECT_EQ(k->at(2).Id(), Tile("m1", 2).Id());
    EXPECT_EQ(k->at(3).Id(), Tile("m1", 3).Id());
    EXPECT_EQ(k->stolen().Type(), tile_type::m1);
    EXPECT_EQ(k->stolen().Id(), Tile("m1", 2).Id());
    EXPECT_EQ(k->tiles().size(), 4);
    EXPECT_TRUE(vec_type_eq(p->tiles(), {tt::m1, tt::m1, tt::m1, tt::m1}));
    EXPECT_EQ(k->tiles_from_hand().size(), 3);
    EXPECT_TRUE(vec_type_eq(p->tiles(), {tt::m1, tt::m1, tt::m1}));
    EXPECT_EQ(k->last().Type(), tile_type::m1);
    EXPECT_EQ(k->last().Id(), Tile("m1", 0).Id());
    EXPECT_EQ(k->undiscardable_tile_types().size(), 0);
}

TEST(open, OpenGenerator)
{
    auto check = [] (Open *o, std::vector<tt> v, tt t, relative_pos f) {
        auto tiles = o -> tiles();
        for (int i = 0; i < v.size(); ++i)
            if (o->at(i).Type() != v.at(i)) return false;
        if (o->stolen().Type() != t) return false;
        if (o->from() != f) return false;
        return true;
    };
    auto og = OpenGenerator();
    std::unique_ptr<Open> o;
    // http://tenhou.net/5/?log=2019101219gm-0009-0000-a1515896
    // EAST 1 0
    o = og.generate(47723);
    EXPECT_EQ(o->type(), open_type::pon);
    EXPECT_TRUE(check(o.get(), {tt::wd, tt::wd, tt::wd}, tt::wd, relative_pos::left));
    o = og.generate(51306);
    EXPECT_EQ(o->type(), open_type::pon);
    EXPECT_TRUE(check(o.get(), {tt::rd, tt::rd, tt::rd}, tt::rd, relative_pos::mid));
    o = og.generate(49495);
    EXPECT_EQ(o->type(), open_type::chi);
    EXPECT_TRUE(check(o.get(), {tt::s3, tt::s4, tt::s5}, tt::s3, relative_pos::left));
    o = og.generate(3146);
    EXPECT_EQ(o->type(), open_type::pon);
    EXPECT_TRUE(check(o.get(), {tt::m3, tt::m3, tt::m3}, tt::m3, relative_pos::mid));
    // EAST 1 1
    o = og.generate(42058);
    EXPECT_EQ(o->type(), open_type::pon);
    EXPECT_TRUE(check(o.get(), {tt::ew, tt::ew, tt::ew}, tt::ew, relative_pos::mid));
    o = og.generate(40489);
    EXPECT_EQ(o->type(), open_type::pon);
    EXPECT_TRUE(check(o.get(), {tt::s9, tt::s9, tt::s9}, tt::s9, relative_pos::right));
    o = og.generate(34911);
    EXPECT_EQ(o->type(), open_type::chi);
    EXPECT_TRUE(check(o.get(), {tt::p5, tt::p6, tt::p7}, tt::p6, relative_pos::left));
    o = og.generate(27178);
    EXPECT_EQ(o->type(), open_type::pon);
    EXPECT_TRUE(check(o.get(), {tt::p9, tt::p9, tt::p9}, tt::p9, relative_pos::mid));
    o = og.generate(37063);
    EXPECT_EQ(o->type(), open_type::chi);
    EXPECT_TRUE(check(o.get(), {tt::p6, tt::p7, tt::p8}, tt::p6, relative_pos::left));
    // East 1 2
    o = og.generate(12905);
    EXPECT_EQ(o->type(), open_type::pon);
    EXPECT_TRUE(check(o.get(), {tt::m9, tt::m9, tt::m9}, tt::m9, relative_pos::right));
    o = og.generate(51753);
    EXPECT_EQ(o->type(), open_type::pon);
    EXPECT_TRUE(check(o.get(), {tt::rd, tt::rd, tt::rd}, tt::rd, relative_pos::right));
    o = og.generate(50679);
    EXPECT_EQ(o->type(), open_type::chi);
    EXPECT_TRUE(check(o.get(), {tt::s3, tt::s4, tt::s5}, tt::s4, relative_pos::left));
    o = og.generate(14679);
    EXPECT_EQ(o->type(), open_type::chi);
    EXPECT_TRUE(check(o.get(), {tt::m5, tt::m6, tt::m7}, tt::m7, relative_pos::left));
    // East 1 3
    o = og.generate(43);
    EXPECT_EQ(o->type(), open_type::pon);
    EXPECT_TRUE(check(o.get(), {tt::m1, tt::m1, tt::m1}, tt::m1, relative_pos::left));
    o = og.generate(7583);
    EXPECT_EQ(o->type(), open_type::chi);
    EXPECT_TRUE(check(o.get(), {tt::m3, tt::m4, tt::m5}, tt::m4, relative_pos::left));
    // East 1 4
    o = og.generate(4649);
    EXPECT_EQ(o->type(), open_type::pon);
    EXPECT_TRUE(check(o.get(), {tt::m4, tt::m4, tt::m4}, tt::m4, relative_pos::right));
    o = og.generate(2063);
    EXPECT_EQ(o->type(), open_type::chi);
    EXPECT_TRUE(check(o.get(), {tt::m1, tt::m2, tt::m3}, tt::m3, relative_pos::left));
    o = og.generate(20615);
    EXPECT_EQ(o->type(), open_type::chi);
    EXPECT_TRUE(check(o.get(), {tt::m7, tt::m8, tt::m9}, tt::m9, relative_pos::left));
    // East 2 0
    o = og.generate(42539);
    EXPECT_EQ(o->type(), open_type::pon);
    EXPECT_TRUE(check(o.get(), {tt::ew, tt::ew, tt::ew}, tt::ew, relative_pos::left));
    // East 2 1
    o = og.generate(46633);
    EXPECT_EQ(o->type(), open_type::pon);
    EXPECT_TRUE(check(o.get(), {tt::nw, tt::nw, tt::nw}, tt::nw, relative_pos::right));
    o = og.generate(41481);
    EXPECT_EQ(o->type(), open_type::pon);
    EXPECT_TRUE(check(o.get(), {tt::ew, tt::ew, tt::ew}, tt::ew, relative_pos::right));
    o = og.generate(31241);
    EXPECT_EQ(o->type(), open_type::pon);
    EXPECT_TRUE(check(o.get(), {tt::s3, tt::s3, tt::s3}, tt::s3, relative_pos::right));
    // East 2 2
    // East 2 3
    o = og.generate(47690);
    EXPECT_EQ(o->type(), open_type::pon);
    EXPECT_TRUE(check(o.get(), {tt::wd, tt::wd, tt::wd}, tt::wd, relative_pos::mid));
    // East 2 4
    o = og.generate(48683);
    EXPECT_EQ(o->type(), open_type::pon);
    EXPECT_TRUE(check(o.get(), {tt::wd, tt::wd, tt::wd}, tt::wd, relative_pos::left));
    o = og.generate(39399);
    EXPECT_EQ(o->type(), open_type::chi);
    EXPECT_TRUE(check(o.get(), {tt::p6, tt::p7, tt::p8}, tt::p8, relative_pos::left));
    o = og.generate(52303);
    EXPECT_EQ(o->type(), open_type::chi);
    EXPECT_TRUE(check(o.get(), {tt::s4, tt::s5, tt::s6}, tt::s4, relative_pos::left));
    // EAST 3 0
    o = og.generate(43081);
    EXPECT_EQ(o->type(), open_type::pon);
    EXPECT_TRUE(check(o.get(), {tt::sw, tt::sw, tt::sw}, tt::sw, relative_pos::right));
    o = og.generate(42058);
    EXPECT_EQ(o->type(), open_type::pon);
    EXPECT_TRUE(check(o.get(), {tt::ew, tt::ew, tt::ew}, tt::ew, relative_pos::mid));
    // EAST 4 0
    o = og.generate(60751);
    EXPECT_EQ(o->type(), open_type::chi);
    EXPECT_TRUE(check(o.get(), {tt::s6, tt::s7, tt::s8}, tt::s8, relative_pos::left));
    o = og.generate(47625);
    EXPECT_EQ(o->type(), open_type::pon);
    EXPECT_TRUE(check(o.get(), {tt::wd, tt::wd, tt::wd}, tt::wd, relative_pos::right));
    // SOUTH 1 0
    o = og.generate(26187);
    EXPECT_EQ(o->type(), open_type::pon);
    EXPECT_TRUE(check(o.get(), {tt::p9, tt::p9, tt::p9}, tt::p9, relative_pos::left));
    o = og.generate(49770);
    EXPECT_EQ(o->type(), open_type::pon);
    EXPECT_TRUE(check(o.get(), {tt::gd, tt::gd, tt::gd}, tt::gd, relative_pos::mid));
    // SOUTH 2 0
    o = og.generate(36459);
    EXPECT_EQ(o->type(), open_type::pon);
    EXPECT_TRUE(check(o.get(), {tt::s6, tt::s6, tt::s6}, tt::s6, relative_pos::left));
    // TODO: add tests from tenhou log for kans
}

