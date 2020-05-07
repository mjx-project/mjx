#include <array>
#include "gtest/gtest.h"
#include "tile.h"
#include "hand.h"
#include "win_cache.h"

using namespace mj;

TEST(hand, Hand)
{
    using tt = tile_type;
    EXPECT_NO_FATAL_FAILURE(
            Hand({tt::m1, tt::m9, tt::p1, tt::p9, tt::s1, tt::s9, tt::ew, tt::sw, tt::ww, tt::nw, tt::wd, tt::gd, tt::rd})
    );
    EXPECT_NO_FATAL_FAILURE(
            Hand({0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60});
    );
    EXPECT_NO_FATAL_FAILURE(
            Hand({"m1", "m9", "p1", "p9", "s1", "s9", "ew", "sw", "ww", "nw", "wd", "gd", "rd"})
    );
    EXPECT_NO_FATAL_FAILURE(
            Hand(Tile::create({"m1", "m9", "p1", "p9", "s1", "s9", "ew", "sw", "ww", "nw", "wd", "gd", "rd"}))
    );
    auto tiles = Tile::create({"m1", "m9", "p1", "p9", "s1", "s9", "ew", "sw", "ww", "nw", "wd", "gd", "rd"});
    EXPECT_NO_FATAL_FAILURE(
            Hand(tiles.begin(), tiles.end())
    );
}

TEST(hand, has)
{
    using tt = tile_type;
    auto h = Hand({"m1", "m1", "p1", "p2", "p3", "s9", "ew", "sw", "ww", "nw", "wd", "gd", "rd"});
    EXPECT_TRUE(h.has({tt::m1}));
    EXPECT_TRUE(h.has({tt::m1, tt::m1}));
    EXPECT_FALSE(h.has({tt::m1, tt::m1, tt::m1}));
    EXPECT_TRUE(h.has({tt::p1, tt::p2, tt::p3}));
    EXPECT_FALSE(h.has({tt::p1, tt::p2, tt::p3, tt::p4}));
}

TEST(hand, draw) {
    auto h = Hand({"m1", "m9", "p1", "p9", "s1", "s9", "ew", "sw", "ww", "nw", "wd", "gd", "rd"});
    EXPECT_EQ(h.size(), 13);
    h.draw(Tile(1));
    EXPECT_EQ(h.phase(), hand_phase::after_draw);
    EXPECT_EQ(h.size(), 14);
}

TEST(hand, chow) {
    auto h = Hand({"m1", "m1", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "m9", "m9"});
    std::vector<Tile> t = {Tile("m2"), Tile("m3"), Tile("m4", 3)};
    auto c = std::make_unique<Chow>(t, Tile("m4", 3));
    EXPECT_EQ(h.phase(), hand_phase::after_discard);
    EXPECT_EQ(h.size(), 13);
    h.chow(std::move(c));
    EXPECT_EQ(h.phase(), hand_phase::after_chow);
    EXPECT_EQ(h.size(), 14);
    EXPECT_EQ(h.size_opened(), 3);
    EXPECT_EQ(h.size_closed(), 11);
    auto possible_discards = h.possible_discards();
    EXPECT_EQ(possible_discards.size(), 7);
    EXPECT_EQ(std::find_if(possible_discards.begin(), possible_discards.end(),
                           [](auto x) { return x.is(tile_type::m4); }), possible_discards.end());
    EXPECT_EQ(std::find_if(possible_discards.begin(), possible_discards.end(),
                           [](auto x) { return x.is(tile_type::m1); }), possible_discards.end());
    EXPECT_NE(std::find_if(possible_discards.begin(), possible_discards.end(),
                           [](auto x) { return x.is(tile_type::m5); }), possible_discards.end());
}

TEST(hand, pung)
{
    auto h = Hand({"m1", "m1", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "m9", "m9"});
    auto p = std::make_unique<Pung>(Tile("m9", 3), Tile("m9", 0), relative_pos::left);
    EXPECT_EQ(h.phase(), hand_phase::after_discard);
    EXPECT_EQ(h.size(), 13);
    h.pung(std::move(p));
    EXPECT_EQ(h.phase(), hand_phase::after_pung);
    EXPECT_EQ(h.size(), 14);
    EXPECT_EQ(h.size_opened(), 3);
    EXPECT_EQ(h.size_closed(), 11);
    auto possible_discards = h.possible_discards();
    EXPECT_EQ(possible_discards.size(), 10);
    EXPECT_EQ(std::find_if(possible_discards.begin(), possible_discards.end(),
                           [](auto x){ return x.is(tile_type::m9); }), possible_discards.end());
    EXPECT_NE(std::find_if(possible_discards.begin(), possible_discards.end(),
                           [](auto x){ return x.is(tile_type::m5); }), possible_discards.end());
}

TEST(hand, kong_mld)
{
    auto h = Hand({"m1", "m1", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "m9", "m9"});
    auto k = std::make_unique<KongMld>(Tile("m9", 3), relative_pos::mid);
    EXPECT_EQ(h.phase(), hand_phase::after_discard);
    EXPECT_EQ(h.size(), 13);
    h.kong_mld(std::move(k));
    EXPECT_EQ(h.phase(), hand_phase::after_kong_mld);
    EXPECT_EQ(h.size(), 14);
    EXPECT_EQ(h.size_opened(), 4);
    EXPECT_EQ(h.size_closed(), 10);
    auto possible_discards = h.possible_discards();
    EXPECT_EQ(possible_discards.size(), 10);
}

TEST(hand, kong_cnc)
{
    auto h = Hand({"m1", "m1", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "m9", "m9"});
    h.draw(Tile("m9", 3));
    auto k = std::make_unique<KongCnc>(Tile("m9", 0));
    EXPECT_EQ(h.phase(), hand_phase::after_draw);
    EXPECT_EQ(h.size(), 14);
    h.kong_cnc(std::move(k));
    EXPECT_EQ(h.phase(), hand_phase::after_kong_cnc);
    EXPECT_EQ(h.size(), 14);
    EXPECT_EQ(h.size_opened(), 4);
    EXPECT_EQ(h.size_closed(), 10);
    auto possible_discards = h.possible_discards();
    EXPECT_EQ(possible_discards.size(), 10);
}

TEST(hand, kong_ext)
{
    auto h = Hand({"m1", "m1", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m8", "m9", "m9"});
    auto p = std::make_unique<Pung>(Tile("m9", 2), Tile("m9", 3), relative_pos::left);
    auto k = std::make_unique<KongExt>(p.get());
    EXPECT_EQ(h.size(), 13);
    h.pung(std::move(p));
    EXPECT_EQ(h.size(), 14);
    EXPECT_EQ(h.size_opened(), 3);
    EXPECT_EQ(h.size_closed(), 11);
    h.discard(Tile("m8"));
    EXPECT_EQ(h.size(), 13);
    EXPECT_EQ(h.size_opened(), 3);
    EXPECT_EQ(h.size_closed(), 10);
    h.draw(Tile("m9", 3));
    EXPECT_EQ(h.size(), 14);
    EXPECT_EQ(h.size_opened(), 3);
    EXPECT_EQ(h.size_closed(), 11);
    h.kong_ext(std::move(k));
    EXPECT_EQ(h.size(), 14);
    EXPECT_EQ(h.size_opened(), 4);
    EXPECT_EQ(h.size_closed(), 10);
    EXPECT_EQ(h.phase(), hand_phase::after_kong_ext);
}

TEST(hand, discard)
{
    auto h = Hand({"m1", "m1", "p1", "p2", "p3", "s9", "ew", "sw", "ww", "nw", "wd", "gd", "rd"});
    EXPECT_EQ(h.size(), 13);
    h.draw(Tile("rd", 2));
    EXPECT_EQ(h.size(), 14);
    EXPECT_EQ(h.phase(), hand_phase::after_draw);
    h.discard(Tile("rd"));
    EXPECT_EQ(h.size(), 13);
    EXPECT_EQ(h.phase(), hand_phase::after_discard);
}

TEST(hand, possible_discards) {
    auto h = Hand({"m1", "m2", "m3", "p9", "s1", "s9", "ew", "sw", "ww", "nw", "wd", "gd", "rd"});
    auto t = Tile::create({"m1", "m2", "m3"});
    auto c = std::make_unique<Chow>(t, Tile("m3", 0));
    h.chow(std::move(c));
    auto possible_discards = h.possible_discards();
    EXPECT_EQ(possible_discards.size(), 10);
    EXPECT_EQ(std::find_if(possible_discards.begin(), possible_discards.end(),
            [](auto x){ return x.is(tile_type::m3); }), possible_discards.end());
}

TEST(hand, possible_chows) { // TODO: add more detailed test
    auto h = Hand({"m1", "m1", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "m9", "m9"});
    // [m1]m2m3
    auto opens = h.possible_chows(Tile("m1", 3));
    EXPECT_EQ(opens.size(), 1);
    // m1[m2]m3, [m2]m3m4
    opens = h.possible_chows(Tile("m2", 3));
    EXPECT_EQ(opens.size(), 2);
    // [m3]m4m5, m2[m3]m4, m1m2[m3]
    opens = h.possible_chows(Tile("m3", 3));
    EXPECT_EQ(opens.size(), 3);
    // [m3]m4m5, [m3]m4*m5, m2[m3]m4, m1m2[m3]
    h = Hand({"m1", "m1", "m1", "m2", "m3", "m4", "m5", "m5", "m6", "m7", "m8", "m9", "m9"});
    opens = h.possible_chows(Tile("m3", 3));
    EXPECT_EQ(opens.size(), 4);
    // [m4]m5m6, m3[m4]m5, m2m3[m4]
    h = Hand({"m1", "m1", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "m9", "m9"});
    opens = h.possible_chows(Tile("m4", 3));
    EXPECT_EQ(opens.size(), 3);
    // [m4]m5m6, [m4]*m5m6, m3[m4]m5, m3[m4]*m5, m2m3[m4]
    h = Hand({"m1", "m1", "m1", "m2", "m3", "m4", "m5", "m5", "m6", "m7", "m8", "m9", "m9"});
    opens = h.possible_chows(Tile("m4", 3));
    EXPECT_EQ(opens.size(), 5);
    // [m5]m6m7, m4[m5]m6, m3m4[m5]
    h = Hand({"m1", "m1", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "m9", "m9"});
    opens = h.possible_chows(Tile("m5", 3));
    EXPECT_EQ(opens.size(), 3);
    // [m5]m6m7, m4[m5]m6, m3m4[m5]
    h = Hand({"m1", "m1", "m1", "m2", "m3", "m4", "m5", "m5", "m6", "m7", "m8", "m9", "m9"});
    opens = h.possible_chows(Tile("m5", 3));
    EXPECT_EQ(opens.size(), 3);
    // [m6]m7m8, m5[m6]m7, m4m5[m6]
    h = Hand({"m1", "m1", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "m9", "m9"});
    opens = h.possible_chows(Tile("m6", 3));
    EXPECT_EQ(opens.size(), 3);
    // [m6]m7m8, m5[m6]m7, *m5[m6]m7, m4m5[m6], m4*m5[m6]
    h = Hand({"m1", "m1", "m1", "m2", "m3", "m4", "m5", "m5", "m6", "m7", "m8", "m9", "m9"});
    opens = h.possible_chows(Tile("m6", 3));
    EXPECT_EQ(opens.size(), 5);
    // [m7]m8m9, m6[m7]m8, m5m6[m7]
    h = Hand({"m1", "m1", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "m9", "m9"});
    opens = h.possible_chows(Tile("m7", 3));
    EXPECT_EQ(opens.size(), 3);
    // [m7]m8m9, m6[m7]m8, m5m6[m7], *m5m6[m7]
    h = Hand({"m1", "m1", "m1", "m2", "m3", "m4", "m5", "m5", "m6", "m7", "m8", "m9", "m9"});
    opens = h.possible_chows(Tile("m7", 3));
    EXPECT_EQ(opens.size(), 4);
    // m7[m8]m9, 6m7[m8]
    h = Hand({"m1", "m1", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "m9", "m9"});
    opens = h.possible_chows(Tile("m8", 2));
    EXPECT_EQ(opens.size(), 2);
    // m7m8[m9]
    h = Hand({"m1", "m1", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "m9", "m9"});
    opens = h.possible_chows(Tile("m9", 2));
    EXPECT_EQ(opens.size(), 1);
}

TEST(hand, possible_pungs) {
    auto h = Hand({"m1", "m1", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "m9", "m9"});
    // No pung is expected
    EXPECT_EQ(h.possible_pungs(Tile("m5", 3), relative_pos::mid).size(), 0);
    h = Hand({"m1", "m1", "m1", "m2", "m3", "m5", "m5", "m6", "m7", "m8", "m9", "m9", "m9"});
    // One possible pung is expected
    EXPECT_EQ(h.possible_pungs(Tile("m5", 3), relative_pos::mid).size(), 1);
    EXPECT_EQ((*h.possible_pungs(Tile("m5", 3), relative_pos::mid).begin())->type(), open_type::pung);
    EXPECT_EQ((*h.possible_pungs(Tile("m5", 3), relative_pos::mid).begin())->at(0).type(), tile_type::m5);
    h = Hand({"m1", "m1", "m1", "m2", "m5", "m5", "m5", "m6", "m7", "m8", "m9", "m9", "m9"});
    // Two possible pungs are expected (w/ red 5 and w/o red 5)
    EXPECT_EQ(h.possible_pungs(Tile("m5", 3), relative_pos::mid).size(), 2);
    EXPECT_EQ((*h.possible_pungs(Tile("m5", 3), relative_pos::mid).begin())->at(0).id() % 4, 0);
    EXPECT_EQ((*h.possible_pungs(Tile("m5", 3), relative_pos::mid).begin())->at(1).id() % 4, 1);
    h = Hand({"m1", "m1", "m1", "m2", "m4", "m4", "m4", "m6", "m7", "m8", "m9", "m9", "m9"});
    // One possible pung is expected
    EXPECT_EQ(h.possible_pungs(Tile("m4", 3), relative_pos::mid).size(), 1);
    EXPECT_EQ((*h.possible_pungs(Tile("m4", 3), relative_pos::mid).begin())->at(0).id() % 4, 0);
    EXPECT_EQ((*h.possible_pungs(Tile("m4", 3), relative_pos::mid).begin())->at(1).id() % 4, 1);
}

TEST(hand, possible_kong_mlds) {
    auto h = Hand({"m1", "m1", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "m9", "m9"});
    EXPECT_EQ(h.size(), 13);
    EXPECT_EQ(h.possible_kong_mlds(Tile("m1", 3), relative_pos::mid).size(), 1);
    EXPECT_EQ((*h.possible_kong_mlds(Tile("m1", 3), relative_pos::mid).begin())->type(), open_type::kong_mld);
    EXPECT_EQ((*h.possible_kong_mlds(Tile("m1", 3), relative_pos::mid).begin())->at(0).type(), tile_type::m1);
    EXPECT_EQ((*h.possible_kong_mlds(Tile("m1", 3), relative_pos::mid).begin())->stolen(), Tile("m1", 3));
    EXPECT_EQ((*h.possible_kong_mlds(Tile("m1", 3), relative_pos::mid).begin())->last(), Tile("m1", 3));
    EXPECT_EQ(h.possible_kong_mlds(Tile("m2", 3), relative_pos::mid).size(), 0);
    EXPECT_EQ(h.possible_kong_mlds(Tile("m9", 3), relative_pos::mid).size(), 1);
    EXPECT_EQ((*h.possible_kong_mlds(Tile("m9", 3), relative_pos::mid).begin())->type(), open_type::kong_mld);
    EXPECT_EQ((*h.possible_kong_mlds(Tile("m9", 3), relative_pos::mid).begin())->at(0).type(), tile_type::m9);
    EXPECT_EQ((*h.possible_kong_mlds(Tile("m9", 3), relative_pos::mid).begin())->stolen(), Tile("m9", 3));
    EXPECT_EQ((*h.possible_kong_mlds(Tile("m9", 3), relative_pos::mid).begin())->last(), Tile("m9", 3));
}

TEST(hand, possible_kong_cncs) {
    auto h = Hand({"m1", "m1", "m1", "m2", "m2", "m3", "m4", "m5", "m6", "m7", "m9", "m9", "m9"});
    h.draw(Tile("m9", 3));
    EXPECT_EQ(h.size(), 14);
    EXPECT_EQ(h.possible_kong_cncs().size(), 1);
    EXPECT_EQ((*h.possible_kong_cncs().begin())->type(), open_type::kong_cnc);
    EXPECT_EQ((*h.possible_kong_cncs().begin())->at(0).type(), tile_type::m9);
    EXPECT_EQ((*h.possible_kong_cncs().begin())->stolen(), Tile("m9", 0));
    EXPECT_EQ((*h.possible_kong_cncs().begin())->last(), Tile("m9", 0));
}

TEST(hand, possible_kong_exts) {
    auto h = Hand({"m1", "m1", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "m9", "m9"});
    h.pung(std::make_unique<Pung>(Tile("m9", 3), Tile("m9", 2), relative_pos::mid));
    h.discard(Tile("m1", 0));
    h.draw(Tile("m8", 2));
    EXPECT_EQ(h.size(), 14);
    EXPECT_EQ(h.size_closed(), 11);
    EXPECT_EQ(h.size_opened(), 3);
    EXPECT_EQ(h.possible_kong_exts().size(), 1);
    EXPECT_EQ((*h.possible_kong_exts().begin())->type(), open_type::kong_ext);
    EXPECT_EQ((*h.possible_kong_exts().begin())->at(0).type(), tile_type::m9);
    EXPECT_EQ((*h.possible_kong_exts().begin())->stolen(), Tile("m9", 3));
    EXPECT_EQ((*h.possible_kong_exts().begin())->last(), Tile("m9", 2));
}

TEST(hand, possible_opens_after_others_discard) {
    auto h = Hand({"m2", "m3", "m4", "m4", "m4", "m5", "m5", "m6", "m7", "m8", "m9", "m9", "m9"});
    auto possible_opens = h.possible_opens_after_others_discard(Tile("m4", 3), relative_pos::left);
    // chow [m4]m5m6, [m4]*m5m6, m3[m4]m5, m3[m4]*m5, m2m3[m4]
    // pung m4m4m4
    // kong m4m4m4m4
    EXPECT_EQ(possible_opens.size(), 7);
    EXPECT_EQ(possible_opens.at(0)->type(), open_type::chow);
    EXPECT_EQ(possible_opens.at(0)->at(0).type(), tile_type::m4);
    EXPECT_TRUE(possible_opens.at(0)->at(1).is_red5());
    EXPECT_EQ(possible_opens.at(1)->type(), open_type::chow);
    EXPECT_EQ(possible_opens.at(1)->at(0).type(), tile_type::m4);
    EXPECT_TRUE(!possible_opens.at(1)->at(1).is_red5());
    EXPECT_EQ(possible_opens.at(2)->type(), open_type::chow);
    EXPECT_EQ(possible_opens.at(2)->at(0).type(), tile_type::m3);
    EXPECT_TRUE(possible_opens.at(2)->at(2).is_red5());
    EXPECT_EQ(possible_opens.at(3)->type(), open_type::chow);
    EXPECT_EQ(possible_opens.at(3)->at(0).type(), tile_type::m3);
    EXPECT_TRUE(!possible_opens.at(3)->at(2).is_red5());
    EXPECT_EQ(possible_opens.at(4)->type(), open_type::chow);
    EXPECT_EQ(possible_opens.at(4)->at(0).type(), tile_type::m2);
    EXPECT_EQ(possible_opens.at(5)->type(), open_type::pung);
    EXPECT_EQ(possible_opens.at(6)->type(), open_type::kong_mld);
}

TEST(hand, possible_opens_after_draw) {
    auto h = Hand({"m1", "m1", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "m9", "m9"});
    h.pung(std::make_unique<Pung>(Tile("m9", 3), Tile("m9", 2), relative_pos::mid));
    h.discard(Tile("m3", 0));
    h.draw(Tile("m1", 3));
    auto possible_opens = h.possible_opens_after_draw();
    EXPECT_EQ(possible_opens.size(), 2);
    EXPECT_EQ(possible_opens.at(0)->type(), open_type::kong_cnc);
    EXPECT_EQ(possible_opens.at(0)->at(0).type(), tile_type::m1);
    EXPECT_EQ(possible_opens.at(1)->type(), open_type::kong_ext);
    EXPECT_EQ(possible_opens.at(1)->at(0).type(), tile_type::m9);
}

TEST(hand, size) {
    auto h = Hand({"m1", "m9", "p1", "p9", "s1", "s9", "ew", "sw", "ww", "nw", "wd", "gd", "rd"});
    EXPECT_EQ(h.size(), 13);
    EXPECT_EQ(h.size_closed(), 13);
    EXPECT_EQ(h.size_opened(), 0);
    // TODO : add test cases with melds
}

TEST(hand, to_vector) {
    auto check_vec = [] (const std::vector<Tile> &v1, const std::vector<Tile>&v2) {
        for (std::size_t i = 0; i < v1.size(); ++i)
            if (v1.at(i).type() != v2.at(i).type()) return false;
        return true;
    };

    auto h = Hand({"m1", "m1", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "m9", "m9"});
    auto chows = h.possible_chows(Tile("m2", 1));
    h.chow(std::move(chows.at(0)));
    h.discard(Tile("m9", 2));
    auto pungs = h.possible_pungs(Tile("m1", 3), relative_pos::mid);
    h.pung(std::move(pungs.at(0)));
    h.discard(Tile("m9", 1));
    EXPECT_EQ(h.size(), 13);
    EXPECT_EQ(h.to_vector().size(), 13);
    EXPECT_EQ(h.size_closed(), 7);
    EXPECT_EQ(h.to_vector_closed().size(), 7);
    EXPECT_EQ(h.size_opened(), 6);
    EXPECT_EQ(h.to_vector_opened().size(), 6);
    EXPECT_TRUE(check_vec(h.to_vector(true),
            Tile::create({"m1", "m1", "m1", "m1", "m2", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9"})));
    EXPECT_TRUE(check_vec(h.to_vector_closed(true),
                          Tile::create({"m1", "m2", "m5", "m6", "m7", "m8", "m9"})));
    EXPECT_TRUE(check_vec(h.to_vector_opened(true),
                          Tile::create({"m1", "m1", "m1", "m2", "m3", "m4"})));
}

TEST(hand, to_array) {
    auto check_arr = [] (const std::array<std::uint8_t, 34> &a1, const std::array<std::uint8_t, 34> &a2) {
        for (std::size_t i = 0; i < 34; ++i) {
            if (a1.at(i) != a2.at(i)) return false;
        }
        return true;
    };

    auto h = Hand({"m1", "m1", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "m9", "m9"});
    auto chows = h.possible_chows(Tile("m2", 1));
    h.chow(std::move(chows.at(0)));
    h.discard(Tile("m9", 2));
    auto pungs = h.possible_pungs(Tile("m1", 3), relative_pos::mid);
    h.pung(std::move(pungs.at(0)));
    h.discard(Tile("m9", 1));
    std::array<std::uint8_t, 34> expected =
    {4,2,1,1,1,1,1,1,1,
     0,0,0,0,0,0,0,0,0,
     0,0,0,0,0,0,0,0,0,
     0,0,0,0,0,0,0};
    EXPECT_TRUE(check_arr(h.to_array(), expected));
    expected =
    {1,1,0,0,1,1,1,1,1,
     0,0,0,0,0,0,0,0,0,
     0,0,0,0,0,0,0,0,0,
     0,0,0,0,0,0,0};
    EXPECT_TRUE(check_arr(h.to_array_closed(), expected));
    expected =
    {3,1,1,1,0,0,0,0,0,
     0,0,0,0,0,0,0,0,0,
     0,0,0,0,0,0,0,0,0,
     0,0,0,0,0,0,0};
    EXPECT_TRUE(check_arr(h.to_array_opened(), expected));
}

TEST(hand, is_menzen) {
    // menzen
    auto h = Hand({"m1", "m1", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "m9", "m9"});
    EXPECT_TRUE(h.is_menzen());
    auto chows = h.possible_chows(Tile("m1", 3));
    h.chow(std::move(chows.at(0)));
    EXPECT_FALSE(h.is_menzen());
}

TEST(hand, is_tenpai) {
    // tenpai
    auto h = Hand({"m1", "m1", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "m9", "m9"});
    const auto win_cache = WinningHandCache();
    EXPECT_TRUE(h.is_tenpai(win_cache));
    h = Hand({"m1", "m1", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "m9", "rd"});
    EXPECT_FALSE(h.is_tenpai(win_cache));
}

TEST(hand, can_riichi) {
    const auto win_cache = WinningHandCache();
    auto h = Hand({"m1", "m1", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "m9", "m9"});
    h.draw(Tile("p1"));
    EXPECT_TRUE(h.can_riichi(win_cache));
    h = Hand({"m1", "m1", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "m9", "p9"});
    h.draw(Tile("p1"));
    EXPECT_FALSE(h.can_riichi(win_cache));
    h = Hand({"m1", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "m9", "m9", "m9"});
    auto chows = h.possible_chows(Tile("m1", 2));
    h.chow(std::move(chows.at(0)));
    h.discard(Tile("m9"));
    h.draw(Tile("p1"));
    EXPECT_FALSE(h.can_riichi(win_cache));
}