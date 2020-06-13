#include <array>
#include "gtest/gtest.h"
#include "tile.h"
#include "hand.h"
#include "win_cache.h"

using namespace mj;

TEST(hand, Hand)
{
    using tt = TileType;
    EXPECT_NO_FATAL_FAILURE(
            Hand({tt::kM1, tt::kM9, tt::kP1, tt::kP9, tt::kS1, tt::kS9, tt::kEW, tt::kSW, tt::kWW, tt::kNW, tt::kWD, tt::kGD, tt::kRD})
    );
    EXPECT_NO_FATAL_FAILURE(
            Hand({0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60});
    );
    EXPECT_NO_FATAL_FAILURE(
            Hand({"m1", "m9", "p1", "p9", "s1", "s9", "ew", "sw", "ww", "nw", "wd", "gd", "rd"})
    );
    EXPECT_NO_FATAL_FAILURE(
            Hand(Tile::Create({"m1", "m9", "p1", "p9", "s1", "s9", "ew", "sw", "ww", "nw", "wd", "gd", "rd"}))
    );
    auto tiles = Tile::Create({"m1", "m9", "p1", "p9", "s1", "s9", "ew", "sw", "ww", "nw", "wd", "gd", "rd"});
    EXPECT_NO_FATAL_FAILURE(
            Hand(tiles.begin(), tiles.end())
    );

    auto hand = Hand(
            {"m1", "m2", "m3", "m4", "m5", "m6", "rd", "rd"},  // closed
            {{"m7", "m8", "m9"}},  // chi
            {},  // pon
            {},  // kan_opend
            {},  // kan_closed
            {{"p1", "p1", "p1", "p1"}}  // kan_added
    );
    auto actual = hand.ToVector(true);
    auto expected = Tile::Create({"m1", "m2", "m3", "m4", "m5", "m6", "rd", "rd", "m7", "m8", "m9", "p1", "p1", "p1", "p1"}, true);
    EXPECT_EQ(actual, expected);
    actual = hand.ToVectorClosed(true);
    expected = Tile::Create({"m1", "m2", "m3", "m4", "m5", "m6", "rd", "rd"}, true);
    EXPECT_EQ(actual, expected);
    actual = hand.ToVectorOpened(true);
    expected = Tile::Create({"m7", "m8", "m9", "p1", "p1", "p1", "p1"}, true);
    EXPECT_EQ(actual, expected);

    hand = Hand(
        {"m1", "m2", "m3", "m4", "m5", "m6", "rd", "rd", "wd", "wd", "wd"},  // closed
        {{"m7", "m8", "m9"}},  // chi
        {},  // pon
        {},  // kan_opend
        {},  // kan_closed
        {}  // kan_added
    );
    actual = hand.ToVector(true);
    expected = Tile::Create({{"m1", "m2", "m3", "m4", "m5", "m6", "rd", "rd", "wd", "wd", "wd", "m7", "m8", "m9"}}, true);
    EXPECT_EQ(actual, expected);
    actual = hand.ToVectorClosed(true);
    expected = Tile::Create({"m1", "m2", "m3", "m4", "m5", "m6", "rd", "rd", "wd", "wd", "wd"}, true);
    EXPECT_EQ(actual, expected);
    actual = hand.ToVectorOpened(true);
    expected = Tile::Create({{"m7", "m8", "m9"}}, true);
    EXPECT_EQ(actual, expected);

    hand = Hand(
            {"m4", "m5", "m6", "rd", "rd", "wd", "wd", "wd"},  // closed
            {{"m1", "m2", "m3"}, {"m7", "m8", "m9"}},  // chi
            {},  // pon
            {},  // kan_opend
            {},  // kan_closed
            {}  // kan_added
    );
    actual = hand.ToVector(true);
    expected = Tile::Create({"m4", "m5", "m6", "rd", "rd", "wd", "wd", "wd", "m1", "m2", "m3", "m7", "m8", "m9"}, true);
    EXPECT_EQ(actual, expected);
    actual = hand.ToVectorClosed(true);
    expected = Tile::Create({"m4", "m5", "m6", "rd", "rd", "wd", "wd", "wd"}, true);
    EXPECT_EQ(actual, expected);
    actual = hand.ToVectorOpened(true);
    expected = Tile::Create({"m1", "m2", "m3", "m7", "m8", "m9"}, true);
    EXPECT_EQ(actual, expected);

    hand = Hand(
            {"m1", "m2", "m3", "m4", "m5", "m6", "rd", "rd", "wd", "wd", "wd"},  // closed
            {},  // chi
            {{"p3", "p3", "p3"}},  // pon
            {},  // kan_opend
            {},  // kan_closed
            {}  // kan_added
    );
    actual = hand.ToVector(true);
    expected = Tile::Create({"m1", "m2", "m3", "m4", "m5", "m6", "rd", "rd", "wd", "wd", "wd", "p3", "p3", "p3"}, true);
    EXPECT_EQ(actual, expected);
    actual = hand.ToVectorClosed(true);
    expected = Tile::Create({"m1", "m2", "m3", "m4", "m5", "m6", "rd", "rd", "wd", "wd", "wd"}, true);
    EXPECT_EQ(actual, expected);
    actual = hand.ToVectorOpened(true);
    expected = Tile::Create({"p3", "p3", "p3"}, true);
    EXPECT_EQ(actual, expected);

    hand = Hand(
            {"m1", "m2", "m3", "m4", "m5", "m6", "rd", "rd"},  // closed
            {},  // chi
            {{"p3", "p3", "p3"}, {"wd", "wd", "wd"}},  // pon
            {},  // kan_opend
            {},  // kan_closed
            {}  // kan_added
    );
    actual = hand.ToVector(true);
    expected = Tile::Create({"m1", "m2", "m3", "m4", "m5", "m6", "rd", "rd", "p3", "p3", "p3", "wd", "wd", "wd"}, true);
    EXPECT_EQ(actual, expected);
    actual = hand.ToVectorClosed(true);
    expected = Tile::Create({"m1", "m2", "m3", "m4", "m5", "m6", "rd", "rd"}, true);
    EXPECT_EQ(actual, expected);
    actual = hand.ToVectorOpened(true);
    expected = Tile::Create({"p3", "p3", "p3", "wd", "wd", "wd"}, true);
    EXPECT_EQ(actual, expected);

    hand = Hand(
            {"nw", "nw"},  // closed
            {},  // chi
            {},  // pon
            {{"p3", "p3", "p3", "p3"}, {"wd", "wd", "wd", "wd"}, {"rd", "rd", "rd", "rd"}, {"gd", "gd", "gd", "gd"}},  // kan_opend
            {},  // kan_closed
            {}  // kan_added
    );
    actual = hand.ToVector(true);
    expected = Tile::Create({"nw", "nw", "p3", "p3", "p3", "p3", "wd", "wd", "wd", "wd", "rd", "rd", "rd", "rd", "gd", "gd", "gd", "gd"}, true);
    EXPECT_EQ(actual, expected);
    actual = hand.ToVectorClosed(true);
    expected = Tile::Create(std::vector<std::string>({"nw", "nw"}), true);
    EXPECT_EQ(actual, expected);
    actual = hand.ToVectorOpened(true);
    expected = Tile::Create({"p3", "p3", "p3", "p3", "wd", "wd", "wd", "wd", "rd", "rd", "rd", "rd", "gd", "gd", "gd", "gd"}, true);
    EXPECT_EQ(actual, expected);

    hand = Hand(
            {"nw", "nw"},  // closed
            {},  // chi
            {},  // pon
            {},  // kan_opend
            {{"p3", "p3", "p3", "p3"}, {"wd", "wd", "wd", "wd"}, {"rd", "rd", "rd", "rd"}, {"gd", "gd", "gd", "gd"}},  // kan_closed
            {}  // kan_added
    );
    actual = hand.ToVector(true);
    expected = Tile::Create({"nw", "nw", "p3", "p3", "p3", "p3", "wd", "wd", "wd", "wd", "rd", "rd", "rd", "rd", "gd", "gd", "gd", "gd"}, true);
    EXPECT_EQ(actual, expected);
    actual = hand.ToVectorClosed(true);
    expected = Tile::Create(std::vector<std::string>({"nw", "nw"}), true);
    EXPECT_EQ(actual, expected);
    actual = hand.ToVectorOpened(true);
    expected = Tile::Create({"p3", "p3", "p3", "p3", "wd", "wd", "wd", "wd", "rd", "rd", "rd", "rd", "gd", "gd", "gd", "gd"}, true);
    EXPECT_EQ(actual, expected);

    hand = Hand(
            {"nw", "nw"},  // closed
            {},  // chi
            {},  // pon
            {},  // kan_opened
            {},   // kan_closed
            {{"p3", "p3", "p3", "p3"}, {"wd", "wd", "wd", "wd"}, {"rd", "rd", "rd", "rd"}, {"gd", "gd", "gd", "gd"}}  // kan_added
    );
    actual = hand.ToVector(true);
    expected = Tile::Create({"nw", "nw", "p3", "p3", "p3", "p3", "wd", "wd", "wd", "wd", "rd", "rd", "rd", "rd", "gd", "gd", "gd", "gd"}, true);
    EXPECT_EQ(actual, expected);
    actual = hand.ToVectorClosed(true);
    expected = Tile::Create(std::vector<std::string>({"nw", "nw"}), true);
    EXPECT_EQ(actual, expected);
    actual = hand.ToVectorOpened(true);
    expected = Tile::Create({"p3", "p3", "p3", "p3", "wd", "wd", "wd", "wd", "rd", "rd", "rd", "rd", "gd", "gd", "gd", "gd"}, true);
    EXPECT_EQ(actual, expected);
}

TEST(hand, Has)
{
    using tt = TileType;
    auto h = Hand({"m1", "m1", "p1", "p2", "p3", "s9", "ew", "sw", "ww", "nw", "wd", "gd", "rd"});
    EXPECT_TRUE(h.Has({tt::kM1}));
    EXPECT_TRUE(h.Has({tt::kM1, tt::kM1}));
    EXPECT_FALSE(h.Has({tt::kM1, tt::kM1, tt::kM1}));
    EXPECT_TRUE(h.Has({tt::kP1, tt::kP2, tt::kP3}));
    EXPECT_FALSE(h.Has({tt::kP1, tt::kP2, tt::kP3, tt::kP4}));
}

TEST(hand, Draw) {
    auto h = Hand({"m1", "m9", "p1", "p9", "s1", "s9", "ew", "sw", "ww", "nw", "wd", "gd", "rd"});
    EXPECT_EQ(h.Size(), 13);
    h.Draw(Tile(1));
    EXPECT_EQ(h.Phase(), TilePhase::kAfterDraw);
    EXPECT_EQ(h.Size(), 14);
}

TEST(hand, ApplyChi) {
    auto h = Hand({"m1", "m1", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "m9", "m9"});
    std::vector<Tile> t = {Tile("m2"), Tile("m3"), Tile("m4", 3)};
    auto c = std::make_unique<Chi>(t, Tile("m4", 3));
    EXPECT_EQ(h.Phase(), TilePhase::kAfterDiscards);
    EXPECT_EQ(h.Size(), 13);
    h.ApplyChi(std::move(c));
    EXPECT_EQ(h.Phase(), TilePhase::kAfterChi);
    EXPECT_EQ(h.Size(), 14);
    EXPECT_EQ(h.SizeOpened(), 3);
    EXPECT_EQ(h.SizeClosed(), 11);
    auto possible_discards = h.PossibleDiscards();
    EXPECT_EQ(possible_discards.size(), 7);
    EXPECT_EQ(std::find_if(possible_discards.begin(), possible_discards.end(),
                           [](Tile x) { return x.Is(TileType::kM4); }), possible_discards.end());
    EXPECT_EQ(std::find_if(possible_discards.begin(), possible_discards.end(),
                           [](Tile x) { return x.Is(TileType::kM1); }), possible_discards.end());
    EXPECT_NE(std::find_if(possible_discards.begin(), possible_discards.end(),
                           [](Tile x) { return x.Is(TileType::kM5); }), possible_discards.end());
}

TEST(hand, ApplyPon)
{
    auto h = Hand({"m1", "m1", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "m9", "m9"});
    auto p = std::make_unique<Pon>(Tile("m9", 3), Tile("m9", 0), RelativePos::kLeft);
    EXPECT_EQ(h.Phase(), TilePhase::kAfterDiscards);
    EXPECT_EQ(h.Size(), 13);
    h.ApplyPon(std::move(p));
    EXPECT_EQ(h.Phase(), TilePhase::kAfterPon);
    EXPECT_EQ(h.Size(), 14);
    EXPECT_EQ(h.SizeOpened(), 3);
    EXPECT_EQ(h.SizeClosed(), 11);
    auto possible_discards = h.PossibleDiscards();
    EXPECT_EQ(possible_discards.size(), 10);
    EXPECT_EQ(std::find_if(possible_discards.begin(), possible_discards.end(),
                           [](Tile x){ return x.Is(TileType::kM9); }), possible_discards.end());
    EXPECT_NE(std::find_if(possible_discards.begin(), possible_discards.end(),
                           [](Tile x){ return x.Is(TileType::kM5); }), possible_discards.end());
}

TEST(hand, ApplyKanOpened)
{
    auto h = Hand({"m1", "m1", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "m9", "m9"});
    auto k = std::make_unique<KanOpened>(Tile("m9", 3), RelativePos::kMid);
    EXPECT_EQ(h.Phase(), TilePhase::kAfterDiscards);
    EXPECT_EQ(h.Size(), 13);
    h.ApplyKanOpened(std::move(k));
    EXPECT_EQ(h.Phase(), TilePhase::kAfterKanOpened);
    EXPECT_EQ(h.Size(), 14);
    EXPECT_EQ(h.SizeOpened(), 4);
    EXPECT_EQ(h.SizeClosed(), 10);
    auto possible_discards = h.PossibleDiscards();
    EXPECT_EQ(possible_discards.size(), 10);
}

TEST(hand, ApplyKanClosed)
{
    auto h = Hand({"m1", "m1", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "m9", "m9"});
    h.Draw(Tile("m9", 3));
    auto k = std::make_unique<KanClosed>(Tile("m9", 0));
    EXPECT_EQ(h.Phase(), TilePhase::kAfterDraw);
    EXPECT_EQ(h.Size(), 14);
    h.ApplyKanClosed(std::move(k));
    EXPECT_EQ(h.Phase(), TilePhase::kAfterKanClosed);
    EXPECT_EQ(h.Size(), 14);
    EXPECT_EQ(h.SizeOpened(), 4);
    EXPECT_EQ(h.SizeClosed(), 10);
    auto possible_discards = h.PossibleDiscards();
    EXPECT_EQ(possible_discards.size(), 10);
}

TEST(hand, ApplyKanAdded)
{
    auto h = Hand({"m1", "m1", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m8", "m9", "m9"});
    auto p = std::make_unique<Pon>(Tile("m9", 2), Tile("m9", 3), RelativePos::kLeft);
    auto k = std::make_unique<KanAdded>(p.get());
    EXPECT_EQ(h.Size(), 13);
    h.ApplyPon(std::move(p));
    EXPECT_EQ(h.Size(), 14);
    EXPECT_EQ(h.SizeOpened(), 3);
    EXPECT_EQ(h.SizeClosed(), 11);
    h.Discard(Tile("m8"));
    EXPECT_EQ(h.Size(), 13);
    EXPECT_EQ(h.SizeOpened(), 3);
    EXPECT_EQ(h.SizeClosed(), 10);
    h.Draw(Tile("m9", 3));
    EXPECT_EQ(h.Size(), 14);
    EXPECT_EQ(h.SizeOpened(), 3);
    EXPECT_EQ(h.SizeClosed(), 11);
    h.ApplyKanAdded(std::move(k));
    EXPECT_EQ(h.Size(), 14);
    EXPECT_EQ(h.SizeOpened(), 4);
    EXPECT_EQ(h.SizeClosed(), 10);
    EXPECT_EQ(h.Phase(), TilePhase::kAfterKanAdded);
}

TEST(hand, Discard)
{
    auto h = Hand({"m1", "m1", "p1", "p2", "p3", "s9", "ew", "sw", "ww", "nw", "wd", "gd", "rd"});
    EXPECT_EQ(h.Size(), 13);
    h.Draw(Tile("rd", 2));
    EXPECT_EQ(h.Size(), 14);
    EXPECT_EQ(h.Phase(), TilePhase::kAfterDraw);
    h.Discard(Tile("rd"));
    EXPECT_EQ(h.Size(), 13);
    EXPECT_EQ(h.Phase(), TilePhase::kAfterDiscards);
}

TEST(hand, PossibleDiscards) {
    auto h = Hand({"m1", "m2", "m3", "p9", "s1", "s9", "ew", "sw", "ww", "nw", "wd", "gd", "rd"});
    auto t = Tile::Create({"m1", "m2", "m3"});
    auto c = std::make_unique<Chi>(t, Tile("m3", 0));
    h.ApplyChi(std::move(c));
    auto possible_discards = h.PossibleDiscards();
    EXPECT_EQ(possible_discards.size(), 10);
    EXPECT_EQ(std::find_if(possible_discards.begin(), possible_discards.end(),
            [](Tile x){ return x.Is(TileType::kM3); }), possible_discards.end());
}

TEST(hand, PossibleOpensAfterOthersDiscard) { // TODO: add more detailed test
    auto num_of_opens = [](const auto &opens, const auto &open_type) {
        return std::count_if(opens.begin(), opens.end(),
                             [&open_type](const auto &x){ return x->Type() == open_type; });
    };

    // Chi
    auto h = Hand({"m1", "m1", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "m9", "m9"});
    // [m1]m2m3
    auto opens = h.PossibleOpensAfterOthersDiscard(Tile("m1", 3), RelativePos::kLeft);
    EXPECT_EQ(num_of_opens(opens, OpenType::kChi), 1);
    // m1[m2]m3, [m2]m3m4
    opens = h.PossibleOpensAfterOthersDiscard(Tile("m2", 3), RelativePos::kLeft);
    EXPECT_EQ(num_of_opens(opens, OpenType::kChi), 2);
    // [m3]m4m5, m2[m3]m4, m1m2[m3]
    opens = h.PossibleOpensAfterOthersDiscard(Tile("m3", 3), RelativePos::kLeft);
    EXPECT_EQ(num_of_opens(opens, OpenType::kChi), 3);
    // [m3]m4m5, [m3]m4*m5, m2[m3]m4, m1m2[m3]
    h = Hand({"m1", "m1", "m1", "m2", "m3", "m4", "m5", "m5", "m6", "m7", "m8", "m9", "m9"});
    opens = h.PossibleOpensAfterOthersDiscard(Tile("m3", 3), RelativePos::kLeft);
    EXPECT_EQ(num_of_opens(opens, OpenType::kChi), 4);
    // [m4]m5m6, m3[m4]m5, m2m3[m4]
    h = Hand({"m1", "m1", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "m9", "m9"});
    opens = h.PossibleOpensAfterOthersDiscard(Tile("m4", 3), RelativePos::kLeft);
    EXPECT_EQ(num_of_opens(opens, OpenType::kChi), 3);
    // [m4]m5m6, [m4]*m5m6, m3[m4]m5, m3[m4]*m5, m2m3[m4]
    h = Hand({"m1", "m1", "m1", "m2", "m3", "m4", "m5", "m5", "m6", "m7", "m8", "m9", "m9"});
    opens = h.PossibleOpensAfterOthersDiscard(Tile("m4", 3), RelativePos::kLeft);
    EXPECT_EQ(num_of_opens(opens, OpenType::kChi), 5);
    // [m5]m6m7, m4[m5]m6, m3m4[m5]
    h = Hand({"m1", "m1", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "m9", "m9"});
    opens = h.PossibleOpensAfterOthersDiscard(Tile("m5", 3), RelativePos::kLeft);
    EXPECT_EQ(num_of_opens(opens, OpenType::kChi), 3);
    // [m5]m6m7, m4[m5]m6, m3m4[m5]
    h = Hand({"m1", "m1", "m1", "m2", "m3", "m4", "m5", "m5", "m6", "m7", "m8", "m9", "m9"});
    opens = h.PossibleOpensAfterOthersDiscard(Tile("m5", 3), RelativePos::kLeft);
    EXPECT_EQ(num_of_opens(opens, OpenType::kChi), 3);
    // [m6]m7m8, m5[m6]m7, m4m5[m6]
    h = Hand({"m1", "m1", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "m9", "m9"});
    opens = h.PossibleOpensAfterOthersDiscard(Tile("m6", 3), RelativePos::kLeft);
    EXPECT_EQ(num_of_opens(opens, OpenType::kChi), 3);
    // [m6]m7m8, m5[m6]m7, *m5[m6]m7, m4m5[m6], m4*m5[m6]
    h = Hand({"m1", "m1", "m1", "m2", "m3", "m4", "m5", "m5", "m6", "m7", "m8", "m9", "m9"});
    opens = h.PossibleOpensAfterOthersDiscard(Tile("m6", 3), RelativePos::kLeft);
    EXPECT_EQ(num_of_opens(opens, OpenType::kChi), 5);
    // [m7]m8m9, m6[m7]m8, m5m6[m7]
    h = Hand({"m1", "m1", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "m9", "m9"});
    opens = h.PossibleOpensAfterOthersDiscard(Tile("m7", 3), RelativePos::kLeft);
    EXPECT_EQ(num_of_opens(opens, OpenType::kChi), 3);
    // [m7]m8m9, m6[m7]m8, m5m6[m7], *m5m6[m7]
    h = Hand({"m1", "m1", "m1", "m2", "m3", "m4", "m5", "m5", "m6", "m7", "m8", "m9", "m9"});
    opens = h.PossibleOpensAfterOthersDiscard(Tile("m7", 3), RelativePos::kLeft);
    EXPECT_EQ(num_of_opens(opens, OpenType::kChi), 4);
    // m7[m8]m9, 6m7[m8]
    h = Hand({"m1", "m1", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "m9", "m9"});
    opens = h.PossibleOpensAfterOthersDiscard(Tile("m8", 2), RelativePos::kLeft);
    EXPECT_EQ(num_of_opens(opens, OpenType::kChi), 2);
    // m7m8[m9]
    h = Hand({"m1", "m1", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "m9", "m9"});
    opens = h.PossibleOpensAfterOthersDiscard(Tile("m9", 2), RelativePos::kLeft);
    EXPECT_EQ(num_of_opens(opens, OpenType::kChi), 1);

    // Pon
    // No pon is expected
    h = Hand({"m1", "m1", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "m9", "m9"});
    opens = h.PossibleOpensAfterOthersDiscard(Tile("m5", 3), RelativePos::kMid);
    EXPECT_EQ(num_of_opens(opens, OpenType::kPon), 0);
    // One possible pon is expected
    h = Hand({"m1", "m1", "m1", "m2", "m3", "m5", "m5", "m6", "m7", "m8", "m9", "m9", "m9"});
    opens = h.PossibleOpensAfterOthersDiscard(Tile("m5", 3), RelativePos::kMid);
    EXPECT_EQ(num_of_opens(opens, OpenType::kPon), 1);
    EXPECT_EQ(opens[0]->Type(), OpenType::kPon);
    EXPECT_EQ(opens[0]->At(0).Type(), TileType::kM5);
    // Two possible pons are expected (w/ red 5 and w/o red 5)
    h = Hand({"m1", "m1", "m1", "m2", "m5", "m5", "m5", "m6", "m7", "m8", "m9", "m9", "m9"});
    opens = h.PossibleOpensAfterOthersDiscard(Tile("m5", 3), RelativePos::kMid);
    EXPECT_EQ(num_of_opens(opens, OpenType::kPon), 2);
    EXPECT_TRUE(opens[0]->At(0).Is(TileType::kM5));
    EXPECT_TRUE(opens[0]->At(0).IsRedFive());
    EXPECT_TRUE(opens[1]->At(0).Is(TileType::kM5));
    EXPECT_FALSE(opens[1]->At(0).IsRedFive());
    // One possible pon is expected
    h = Hand({"m1", "m1", "m1", "m2", "m4", "m4", "m4", "m6", "m7", "m8", "m9", "m9", "m9"});
    opens = h.PossibleOpensAfterOthersDiscard(Tile("m4", 3), RelativePos::kMid);
    EXPECT_EQ(num_of_opens(opens, OpenType::kPon), 1);

    // KanOpened
    h = Hand({"m1", "m1", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "m9", "m9"});
    EXPECT_EQ(h.Size(), 13);
    opens = h.PossibleOpensAfterOthersDiscard(Tile("m1", 3), RelativePos::kMid);
    EXPECT_EQ(opens.size(), 2);
    EXPECT_EQ(num_of_opens(opens, OpenType::kKanOpened), 1);
    EXPECT_EQ(opens.back()->Type(), OpenType::kKanOpened);
    EXPECT_EQ(opens.back()->At(0).Type(), TileType::kM1);
    EXPECT_EQ(opens.back()->StolenTile(), Tile("m1", 3));
    EXPECT_EQ(opens.back()->LastTile(), Tile("m1", 3));
    opens = h.PossibleOpensAfterOthersDiscard(Tile("m2", 3), RelativePos::kMid);
    EXPECT_EQ(num_of_opens(opens, OpenType::kKanOpened), 0);
    opens = h.PossibleOpensAfterOthersDiscard(Tile("m9", 3), RelativePos::kMid);
    EXPECT_EQ(opens.size(), 2);
    EXPECT_EQ(num_of_opens(opens, OpenType::kKanOpened), 1);
    EXPECT_EQ(opens.back()->Type(), OpenType::kKanOpened);
    EXPECT_EQ(opens.back()->At(0).Type(), TileType::kM9);
    EXPECT_EQ(opens.back()->StolenTile(), Tile("m9", 3));
    EXPECT_EQ(opens.back()->LastTile(), Tile("m9", 3));

    // Mixed
    h = Hand({"m2", "m3", "m4", "m4", "m4", "m5", "m5", "m6", "m7", "m8", "m9", "m9", "m9"});
    auto possible_opens = h.PossibleOpensAfterOthersDiscard(Tile("m4", 3), RelativePos::kLeft);
    // chi [m4]m5m6, [m4]*m5m6, m3[m4]m5, m3[m4]*m5, m2m3[m4]
    // pon m4m4m4
    // kan m4m4m4m4
    EXPECT_EQ(possible_opens.size(), 7);
    EXPECT_EQ(possible_opens.at(0)->Type(), OpenType::kChi);
    EXPECT_EQ(possible_opens.at(0)->At(0).Type(), TileType::kM4);
    EXPECT_TRUE(possible_opens.at(0)->At(1).IsRedFive());
    EXPECT_EQ(possible_opens.at(1)->Type(), OpenType::kChi);
    EXPECT_EQ(possible_opens.at(1)->At(0).Type(), TileType::kM4);
    EXPECT_TRUE(!possible_opens.at(1)->At(1).IsRedFive());
    EXPECT_EQ(possible_opens.at(2)->Type(), OpenType::kChi);
    EXPECT_EQ(possible_opens.at(2)->At(0).Type(), TileType::kM3);
    EXPECT_TRUE(possible_opens.at(2)->At(2).IsRedFive());
    EXPECT_EQ(possible_opens.at(3)->Type(), OpenType::kChi);
    EXPECT_EQ(possible_opens.at(3)->At(0).Type(), TileType::kM3);
    EXPECT_TRUE(!possible_opens.at(3)->At(2).IsRedFive());
    EXPECT_EQ(possible_opens.at(4)->Type(), OpenType::kChi);
    EXPECT_EQ(possible_opens.at(4)->At(0).Type(), TileType::kM2);
    EXPECT_EQ(possible_opens.at(5)->Type(), OpenType::kPon);
    EXPECT_EQ(possible_opens.at(6)->Type(), OpenType::kKanOpened);
}

TEST(hand, PossibleKanClosed) {
    auto h = Hand({"m1", "m1", "m1", "m2", "m2", "m3", "m4", "m5", "m6", "m7", "m9", "m9", "m9"});
    h.Draw(Tile("m9", 3));
    EXPECT_EQ(h.Size(), 14);
    EXPECT_EQ(h.PossibleKanClosed().size(), 1);
    EXPECT_EQ((*h.PossibleKanClosed().begin())->Type(), OpenType::kKanClosed);
    EXPECT_EQ((*h.PossibleKanClosed().begin())->At(0).Type(), TileType::kM9);
    EXPECT_EQ((*h.PossibleKanClosed().begin())->StolenTile(), Tile("m9", 0));
    EXPECT_EQ((*h.PossibleKanClosed().begin())->LastTile(), Tile("m9", 0));
}

TEST(hand, PossibleKanAdded) {
    auto h = Hand({"m1", "m1", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "m9", "m9"});
    h.ApplyPon(std::make_unique<Pon>(Tile("m9", 3), Tile("m9", 2), RelativePos::kMid));
    h.Discard(Tile("m1", 0));
    h.Draw(Tile("m8", 2));
    EXPECT_EQ(h.Size(), 14);
    EXPECT_EQ(h.SizeClosed(), 11);
    EXPECT_EQ(h.SizeOpened(), 3);
    EXPECT_EQ(h.PossibleKanAdded().size(), 1);
    EXPECT_EQ((*h.PossibleKanAdded().begin())->Type(), OpenType::kKanAdded);
    EXPECT_EQ((*h.PossibleKanAdded().begin())->At(0).Type(), TileType::kM9);
    EXPECT_EQ((*h.PossibleKanAdded().begin())->StolenTile(), Tile("m9", 3));
    EXPECT_EQ((*h.PossibleKanAdded().begin())->LastTile(), Tile("m9", 2));
}


TEST(hand, PossibleOpensAfterDraw) {
    auto h = Hand({"m1", "m1", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "m9", "m9"});
    h.ApplyPon(std::make_unique<Pon>(Tile("m9", 3), Tile("m9", 2), RelativePos::kMid));
    h.Discard(Tile("m3", 0));
    h.Draw(Tile("m1", 3));
    auto possible_opens = h.PossibleOpensAfterDraw();
    EXPECT_EQ(possible_opens.size(), 2);
    EXPECT_EQ(possible_opens.at(0)->Type(), OpenType::kKanClosed);
    EXPECT_EQ(possible_opens.at(0)->At(0).Type(), TileType::kM1);
    EXPECT_EQ(possible_opens.at(1)->Type(), OpenType::kKanAdded);
    EXPECT_EQ(possible_opens.at(1)->At(0).Type(), TileType::kM9);
}

TEST(hand, Size) {
    auto h = Hand({"m1", "m9", "p1", "p9", "s1", "s9", "ew", "sw", "ww", "nw", "wd", "gd", "rd"});
    EXPECT_EQ(h.Size(), 13);
    EXPECT_EQ(h.SizeClosed(), 13);
    EXPECT_EQ(h.SizeOpened(), 0);
    // TODO : add test cases with melds
}

TEST(hand, ToVector) {
    auto check_vec = [] (const std::vector<Tile> &v1, const std::vector<Tile>&v2) {
        for (std::size_t i = 0; i < v1.size(); ++i)
            if (v1.at(i).Type() != v2.at(i).Type()) return false;
        return true;
    };

    auto h = Hand({"m1", "m1", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "m9", "m9"});
    auto chis = h.PossibleOpensAfterOthersDiscard(Tile("m2", 1), RelativePos::kLeft);
    h.ApplyChi(std::move(chis.at(0)));
    h.Discard(Tile("m9", 2));
    auto pons = h.PossibleOpensAfterOthersDiscard(Tile("m1", 3), RelativePos::kMid);
    h.ApplyPon(std::move(pons.at(0)));
    h.Discard(Tile("m9", 1));
    EXPECT_EQ(h.Size(), 13);
    EXPECT_EQ(h.ToVector().size(), 13);
    EXPECT_EQ(h.SizeClosed(), 7);
    EXPECT_EQ(h.ToVectorClosed().size(), 7);
    EXPECT_EQ(h.SizeOpened(), 6);
    EXPECT_EQ(h.ToVectorOpened().size(), 6);
    EXPECT_TRUE(check_vec(h.ToVector(true),
                          Tile::Create({"m1", "m1", "m1", "m1", "m2", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9"})));
    EXPECT_TRUE(check_vec(h.ToVectorClosed(true),
                          Tile::Create({"m1", "m2", "m5", "m6", "m7", "m8", "m9"})));
    EXPECT_TRUE(check_vec(h.ToVectorOpened(true),
                          Tile::Create({"m1", "m1", "m1", "m2", "m3", "m4"})));
}

TEST(hand, ToArray) {
    auto check_arr = [] (const std::array<std::uint8_t, 34> &a1, const std::array<std::uint8_t, 34> &a2) {
        for (std::size_t i = 0; i < 34; ++i) {
            if (a1.at(i) != a2.at(i)) return false;
        }
        return true;
    };

    auto h = Hand({"m1", "m1", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "m9", "m9"});
    auto chis = h.PossibleOpensAfterOthersDiscard(Tile("m2", 1), RelativePos::kLeft);
    h.ApplyChi(std::move(chis.at(0)));
    h.Discard(Tile("m9", 2));
    auto pons = h.PossibleOpensAfterOthersDiscard(Tile("m1", 3), RelativePos::kMid);
    h.ApplyPon(std::move(pons.at(0)));
    h.Discard(Tile("m9", 1));
    std::array<std::uint8_t, 34> expected =
    {4,2,1,1,1,1,1,1,1,
     0,0,0,0,0,0,0,0,0,
     0,0,0,0,0,0,0,0,0,
     0,0,0,0,0,0,0};
    EXPECT_TRUE(check_arr(h.ToArray(), expected));
    expected =
    {1,1,0,0,1,1,1,1,1,
     0,0,0,0,0,0,0,0,0,
     0,0,0,0,0,0,0,0,0,
     0,0,0,0,0,0,0};
    EXPECT_TRUE(check_arr(h.ToArrayClosed(), expected));
    expected =
    {3,1,1,1,0,0,0,0,0,
     0,0,0,0,0,0,0,0,0,
     0,0,0,0,0,0,0,0,0,
     0,0,0,0,0,0,0};
    EXPECT_TRUE(check_arr(h.ToArrayOpened(), expected));
}

TEST(hand, IsMenzen) {
    // menzen
    auto h = Hand({"m1", "m1", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "m9", "m9"});
    EXPECT_TRUE(h.IsMenzen());
    auto chis = h.PossibleOpensAfterOthersDiscard(Tile("m1", 3), RelativePos::kLeft);
    h.ApplyChi(std::move(chis.at(0)));
    EXPECT_FALSE(h.IsMenzen());
}

TEST(hand, IsTenpai) {
    // tenpai
    auto h = Hand({"m1", "m1", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "m9", "m9"});
    const auto win_cache = WinningHandCache();
    EXPECT_TRUE(h.IsTenpai(win_cache));
    h = Hand({"m1", "m1", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "m9", "rd"});
    EXPECT_FALSE(h.IsTenpai(win_cache));
}

TEST(hand, CanRiichi) {
    const auto win_cache = WinningHandCache();
    auto h = Hand({"m1", "m1", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "m9", "m9"});
    h.Draw(Tile("p1"));
    EXPECT_TRUE(h.CanRiichi(win_cache));
    h = Hand({"m1", "m1", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "m9", "p9"});
    h.Draw(Tile("p1"));
    EXPECT_FALSE(h.CanRiichi(win_cache));
    h = Hand({"m1", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "m9", "m9", "m9"});
    auto chis = h.PossibleOpensAfterOthersDiscard(Tile("m1", 2), RelativePos::kLeft);
    h.ApplyChi(std::move(chis.at(0)));
    h.Discard(Tile("m9"));
    h.Draw(Tile("p1"));
    EXPECT_FALSE(h.CanRiichi(win_cache));
}