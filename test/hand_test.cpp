#include <array>
#include "gtest/gtest.h"
#include "tile.h"
#include "hand.h"
#include "win_cache.h"

using namespace mj;

TEST(hand, Hand)
{
    EXPECT_NO_FATAL_FAILURE(
            Hand(Tile::Create({"m1", "m9", "p1", "p9", "s1", "s9", "ew", "sw", "ww", "nw", "wd", "gd", "rd"}))
    );
    auto tiles = Tile::Create({"m1", "m9", "p1", "p9", "s1", "s9", "ew", "sw", "ww", "nw", "wd", "gd", "rd"});
    EXPECT_NO_FATAL_FAILURE(
            Hand(tiles.begin(), tiles.end())
    );

    auto hand = Hand(HandParams("m1,m2,m3,m4,m5,rd,rd").Chi("m7,m8,m9").KanAdded("p1,p1,p1,p1"));
    auto actual = hand.ToVector(true);
    auto expected = Tile::Create({"m1", "m2", "m3", "m4", "m5", "rd", "rd", "m7", "m8", "m9", "p1", "p1", "p1", "p1"}, true);
    EXPECT_EQ(actual, expected);
    actual = hand.ToVectorClosed(true);
    expected = Tile::Create({"m1", "m2", "m3", "m4", "m5", "rd", "rd"}, true);
    EXPECT_EQ(actual, expected);
    actual = hand.ToVectorOpened(true);
    expected = Tile::Create({"m7", "m8", "m9", "p1", "p1", "p1", "p1"}, true);
    EXPECT_EQ(actual, expected);

    hand = Hand(HandParams("m1,m2,m3,m4,m5,wd,wd,wd,rd,rd").Chi("m7,m8,m9"));
    actual = hand.ToVector(true);
    expected = Tile::Create({{"m1", "m2", "m3", "m4", "m5", "rd", "rd", "wd", "wd", "wd", "m7", "m8", "m9"}}, true);
    EXPECT_EQ(actual, expected);
    actual = hand.ToVectorClosed(true);
    expected = Tile::Create({"m1", "m2", "m3", "m4", "m5", "rd", "rd", "wd", "wd", "wd"}, true);
    EXPECT_EQ(actual, expected);
    actual = hand.ToVectorOpened(true);
    expected = Tile::Create({{"m7", "m8", "m9"}}, true);
    EXPECT_EQ(actual, expected);

    hand = Hand(HandParams("m4,m5,rd,rd,wd,wd,wd").Chi("m1,m2,m3").Chi("m7,m8,m9"));
    actual = hand.ToVector(true);
    expected = Tile::Create({"m4", "m5", "rd", "rd", "wd", "wd", "wd", "m1", "m2", "m3", "m7", "m8", "m9"}, true);
    EXPECT_EQ(actual, expected);
    actual = hand.ToVectorClosed(true);
    expected = Tile::Create({"m4", "m5", "rd", "rd", "wd", "wd", "wd"}, true);
    EXPECT_EQ(actual, expected);
    actual = hand.ToVectorOpened(true);
    expected = Tile::Create({"m1", "m2", "m3", "m7", "m8", "m9"}, true);
    EXPECT_EQ(actual, expected);

    hand = Hand(HandParams("m1,m2,m3,m4,m5,rd,rd,wd,wd,wd").Pon("p3,p3,p3"));
    actual = hand.ToVector(true);
    expected = Tile::Create({"m1", "m2", "m3", "m4", "m5", "rd", "rd", "wd", "wd", "wd", "p3", "p3", "p3"}, true);
    EXPECT_EQ(actual, expected);
    actual = hand.ToVectorClosed(true);
    expected = Tile::Create({"m1", "m2", "m3", "m4", "m5", "rd", "rd", "wd", "wd", "wd"}, true);
    EXPECT_EQ(actual, expected);
    actual = hand.ToVectorOpened(true);
    expected = Tile::Create({"p3", "p3", "p3"}, true);
    EXPECT_EQ(actual, expected);

    hand = Hand(HandParams("m1,m2,m3,m4,m5,rd,rd").Pon("p3,p3,p3").Pon("wd,wd,wd"));
    actual = hand.ToVector(true);
    expected = Tile::Create({"m1", "m2", "m3", "m4", "m5", "rd", "rd", "p3", "p3", "p3", "wd", "wd", "wd"}, true);
    EXPECT_EQ(actual, expected);
    actual = hand.ToVectorClosed(true);
    expected = Tile::Create({"m1", "m2", "m3", "m4", "m5", "rd", "rd"}, true);
    EXPECT_EQ(actual, expected);
    actual = hand.ToVectorOpened(true);
    expected = Tile::Create({"p3", "p3", "p3", "wd", "wd", "wd"}, true);
    EXPECT_EQ(actual, expected);

    hand = Hand(HandParams("nw").KanOpened("p3,p3,p3,p3").KanOpened("wd,wd,wd,wd").KanOpened("rd,rd,rd,rd").KanOpened("gd,gd,gd,gd"));
    actual = hand.ToVector(true);
    expected = Tile::Create({"nw", "p3", "p3", "p3", "p3", "wd", "wd", "wd", "wd", "rd", "rd", "rd", "rd", "gd", "gd", "gd", "gd"}, true);
    EXPECT_EQ(actual, expected);
    actual = hand.ToVectorClosed(true);
    expected = Tile::Create(std::vector<std::string>({"nw"}), true);
    EXPECT_EQ(actual, expected);
    actual = hand.ToVectorOpened(true);
    expected = Tile::Create({"p3", "p3", "p3", "p3", "wd", "wd", "wd", "wd", "rd", "rd", "rd", "rd", "gd", "gd", "gd", "gd"}, true);
    EXPECT_EQ(actual, expected);

    hand = Hand(HandParams("nw").KanClosed("p3,p3,p3,p3").KanClosed("wd,wd,wd,wd").KanClosed("rd,rd,rd,rd").KanClosed("gd,gd,gd,gd"));
    actual = hand.ToVector(true);
    expected = Tile::Create({"nw", "p3", "p3", "p3", "p3", "wd", "wd", "wd", "wd", "rd", "rd", "rd", "rd", "gd", "gd", "gd", "gd"}, true);
    EXPECT_EQ(actual, expected);
    actual = hand.ToVectorClosed(true);
    expected = Tile::Create(std::vector<std::string>({"nw"}), true);
    EXPECT_EQ(actual, expected);
    actual = hand.ToVectorOpened(true);
    expected = Tile::Create({"p3", "p3", "p3", "p3", "wd", "wd", "wd", "wd", "rd", "rd", "rd", "rd", "gd", "gd", "gd", "gd"}, true);
    EXPECT_EQ(actual, expected);

    hand = Hand(HandParams("nw").KanAdded("p3,p3,p3,p3").KanAdded("wd,wd,wd,wd").KanAdded("rd,rd,rd,rd").KanAdded("gd,gd,gd,gd"));
    actual = hand.ToVector(true);
    expected = Tile::Create({"nw","p3", "p3", "p3", "p3", "wd", "wd", "wd", "wd", "rd", "rd", "rd", "rd", "gd", "gd", "gd", "gd"}, true);
    EXPECT_EQ(actual, expected);
    actual = hand.ToVectorClosed(true);
    expected = Tile::Create(std::vector<std::string>({"nw"}), true);
    EXPECT_EQ(actual, expected);
    actual = hand.ToVectorOpened(true);
    expected = Tile::Create({"p3", "p3", "p3", "p3", "wd", "wd", "wd", "wd", "rd", "rd", "rd", "rd", "gd", "gd", "gd", "gd"}, true);
    EXPECT_EQ(actual, expected);

    hand = Hand(HandParams("m1,m2,m3,m4,m5,rd,rd,m7,m8,m9").KanClosed("p1,p1,p1,p1").Riichi().Tsumo("m6"));
    EXPECT_TRUE(hand.IsMenzen());
    EXPECT_TRUE(hand.IsUnderRiichi());
}

TEST(hand, Draw) {
    auto h = Hand(HandParams("m1,m9,p1,p9,s1,s9,ew,sw,ww,nw,wd,gd,rd"));
    EXPECT_EQ(h.Size(), 13);
    h.Draw(Tile(1));
    EXPECT_EQ(h.Stage(), HandStage::kAfterDraw);
    EXPECT_EQ(h.Size(), 14);
}

TEST(hand, ApplyChi) {
    auto h = Hand(HandParams("m1,m1,m1,m2,m3,m4,m5,m6,m7,m8,m9,m9,m9"));
    std::vector<Tile> t = {Tile("m2"), Tile("m3"), Tile("m4", 3)};
    auto c = std::make_unique<Chi>(t, Tile("m4", 3));
    EXPECT_EQ(h.Stage(), HandStage::kAfterDiscards);
    EXPECT_EQ(h.Size(), 13);
    h.ApplyOpen(std::move(c));
    EXPECT_EQ(h.Stage(), HandStage::kAfterChi);
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
    auto h = Hand(HandParams("m1,m1,m1,m2,m3,m4,m5,m6,m7,m8,m9,m9,m9"));
    auto p = std::make_unique<Pon>(Tile("m9", 3), Tile("m9", 0), RelativePos::kLeft);
    EXPECT_EQ(h.Stage(), HandStage::kAfterDiscards);
    EXPECT_EQ(h.Size(), 13);
    h.ApplyOpen(std::move(p));
    EXPECT_EQ(h.Stage(), HandStage::kAfterPon);
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
    auto h = Hand(HandParams("m1,m1,m1,m2,m3,m4,m5,m6,m7,m8,m9,m9,m9"));
    auto k = std::make_unique<KanOpened>(Tile("m9", 3), RelativePos::kMid);
    EXPECT_EQ(h.Stage(), HandStage::kAfterDiscards);
    EXPECT_EQ(h.Size(), 13);
    h.ApplyOpen(std::move(k));
    EXPECT_EQ(h.Stage(), HandStage::kAfterKanOpened);
    EXPECT_EQ(h.Size(), 14);
    EXPECT_EQ(h.SizeOpened(), 4);
    EXPECT_EQ(h.SizeClosed(), 10);
    h.Draw(Tile("m3", 3));
    auto possible_discards = h.PossibleDiscards();
    EXPECT_EQ(possible_discards.size(), 11);
}

TEST(hand, ApplyKanClosed)
{
    auto h = Hand(HandParams("m1,m1,m1,m2,m3,m4,m5,m6,m7,m8,m9,m9,m9"));
    h.Draw(Tile("m9", 3));
    auto k = std::make_unique<KanClosed>(Tile("m9", 0));
    EXPECT_EQ(h.Stage(), HandStage::kAfterDraw);
    EXPECT_EQ(h.Size(), 14);
    h.ApplyOpen(std::move(k));
    EXPECT_EQ(h.Stage(), HandStage::kAfterKanClosed);
    EXPECT_EQ(h.Size(), 14);
    EXPECT_EQ(h.SizeOpened(), 4);
    EXPECT_EQ(h.SizeClosed(), 10);
    h.Draw(Tile("m3", 3));
    auto possible_discards = h.PossibleDiscards();
    EXPECT_EQ(possible_discards.size(), 11);
}

TEST(hand, ApplyKanAdded)
{
    auto h = Hand(HandParams("m1,m1,m1,m2,m3,m4,m5,m6,m7,m8,m8,m9,m9"));
    auto p = std::make_unique<Pon>(Tile("m9", 2), Tile("m9", 3), RelativePos::kLeft);
    auto k = std::make_unique<KanAdded>(p.get());
    EXPECT_EQ(h.Size(), 13);
    h.ApplyOpen(std::move(p));
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
    h.ApplyOpen(std::move(k));
    EXPECT_EQ(h.Size(), 14);
    EXPECT_EQ(h.SizeOpened(), 4);
    EXPECT_EQ(h.SizeClosed(), 10);
    EXPECT_EQ(h.Stage(), HandStage::kAfterKanAdded);
}

TEST(hand, Discard)
{
    auto h = Hand(HandParams("m1,m1,p1,p2,p3,s9,ew,sw,ww,nw,wd,gd,rd"));
    EXPECT_EQ(h.Size(), 13);
    h.Draw(Tile("rd", 2));
    EXPECT_EQ(h.Size(), 14);
    EXPECT_EQ(h.Stage(), HandStage::kAfterDraw);
    h.Discard(Tile("rd"));
    EXPECT_EQ(h.Size(), 13);
    EXPECT_EQ(h.Stage(), HandStage::kAfterDiscards);
}

TEST(hand, PossibleDiscards) {
    auto h = Hand(HandParams("m1,m2,m3,p9,s1,s9,ew,sw,ww,nw,wd,gd,rd"));
    auto t = Tile::Create({"m1", "m2", "m3"});
    auto c = std::make_unique<Chi>(t, Tile("m3", 0));
    h.ApplyOpen(std::move(c));
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
    auto h = Hand(HandParams("m1,m1,m1,m2,m3,m4,m5,m6,m7,m8,m9,m9,m9"));
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
    h = Hand(HandParams("m1,m1,m1,m2,m3,m4,m5,m5,m6,m7,m8,m9,m9"));
    opens = h.PossibleOpensAfterOthersDiscard(Tile("m3", 3), RelativePos::kLeft);
    EXPECT_EQ(num_of_opens(opens, OpenType::kChi), 4);
    // [m4]m5m6, m3[m4]m5, m2m3[m4]
    h = Hand(HandParams("m1,m1,m1,m2,m3,m4,m5,m6,m7,m8,m9,m9,m9"));
    opens = h.PossibleOpensAfterOthersDiscard(Tile("m4", 3), RelativePos::kLeft);
    EXPECT_EQ(num_of_opens(opens, OpenType::kChi), 3);
    // [m4]m5m6, [m4]*m5m6, m3[m4]m5, m3[m4]*m5, m2m3[m4]
    h = Hand(HandParams("m1,m1,m1,m2,m3,m4,m5,m5,m6,m7,m8,m9,m9"));
    opens = h.PossibleOpensAfterOthersDiscard(Tile("m4", 3), RelativePos::kLeft);
    EXPECT_EQ(num_of_opens(opens, OpenType::kChi), 5);
    // [m5]m6m7, m4[m5]m6, m3m4[m5]
    h = Hand(HandParams("m1,m1,m1,m2,m3,m4,m5,m6,m7,m8,m9,m9,m9"));
    opens = h.PossibleOpensAfterOthersDiscard(Tile("m5", 3), RelativePos::kLeft);
    EXPECT_EQ(num_of_opens(opens, OpenType::kChi), 3);
    // [m5]m6m7, m4[m5]m6, m3m4[m5]
    h = Hand(HandParams("m1,m1,m1,m2,m3,m4,m5,m5,m6,m7,m8,m9,m9"));
    opens = h.PossibleOpensAfterOthersDiscard(Tile("m5", 3), RelativePos::kLeft);
    EXPECT_EQ(num_of_opens(opens, OpenType::kChi), 3);
    // [m6]m7m8, m5[m6]m7, m4m5[m6]
    h = Hand(HandParams("m1,m1,m1,m2,m3,m4,m5,m6,m7,m8,m9,m9,m9"));
    opens = h.PossibleOpensAfterOthersDiscard(Tile("m6", 3), RelativePos::kLeft);
    EXPECT_EQ(num_of_opens(opens, OpenType::kChi), 3);
    // [m6]m7m8, m5[m6]m7, *m5[m6]m7, m4m5[m6], m4*m5[m6]
    h = Hand(HandParams("m1,m1,m1,m2,m3,m4,m5,m5,m6,m7,m8,m9,m9"));
    opens = h.PossibleOpensAfterOthersDiscard(Tile("m6", 3), RelativePos::kLeft);
    EXPECT_EQ(num_of_opens(opens, OpenType::kChi), 5);
    // [m7]m8m9, m6[m7]m8, m5m6[m7]
    h = Hand(HandParams("m1,m1,m1,m2,m3,m4,m5,m6,m7,m8,m9,m9,m9"));
    opens = h.PossibleOpensAfterOthersDiscard(Tile("m7", 3), RelativePos::kLeft);
    EXPECT_EQ(num_of_opens(opens, OpenType::kChi), 3);
    // [m7]m8m9, m6[m7]m8, m5m6[m7], *m5m6[m7]
    h = Hand(HandParams("m1,m1,m1,m2,m3,m4,m5,m5,m6,m7,m8,m9,m9"));
    opens = h.PossibleOpensAfterOthersDiscard(Tile("m7", 3), RelativePos::kLeft);
    EXPECT_EQ(num_of_opens(opens, OpenType::kChi), 4);
    // m7[m8]m9, 6m7[m8]
    h = Hand(HandParams("m1,m1,m1,m2,m3,m4,m5,m6,m7,m8,m9,m9,m9"));
    opens = h.PossibleOpensAfterOthersDiscard(Tile("m8", 2), RelativePos::kLeft);
    EXPECT_EQ(num_of_opens(opens, OpenType::kChi), 2);
    // m7m8[m9]
    h = Hand(HandParams("m1,m1,m1,m2,m3,m4,m5,m6,m7,m8,m9,m9,m9"));
    opens = h.PossibleOpensAfterOthersDiscard(Tile("m9", 2), RelativePos::kLeft);
    EXPECT_EQ(num_of_opens(opens, OpenType::kChi), 1);

    // Pon
    // No pon is expected
    h = Hand(HandParams("m1,m1,m1,m2,m3,m4,m5,m6,m7,m8,m9,m9,m9"));
    opens = h.PossibleOpensAfterOthersDiscard(Tile("m5", 3), RelativePos::kMid);
    EXPECT_EQ(num_of_opens(opens, OpenType::kPon), 0);
    // One possible pon is expected
    h = Hand(HandParams("m1,m1,m1,m2,m3,m5,m5,m6,m7,m8,m9,m9,m9"));
    opens = h.PossibleOpensAfterOthersDiscard(Tile("m5", 3), RelativePos::kMid);
    EXPECT_EQ(num_of_opens(opens, OpenType::kPon), 1);
    EXPECT_EQ(opens[0]->Type(), OpenType::kPon);
    EXPECT_EQ(opens[0]->At(0).Type(), TileType::kM5);
    // Two possible pons are expected (w/ red 5 and w/o red 5)
    h = Hand(HandParams("m1,m1,m1,m2,m5,m5,m5,m6,m7,m8,m9,m9,m9"));
    opens = h.PossibleOpensAfterOthersDiscard(Tile("m5", 3), RelativePos::kMid);
    EXPECT_EQ(num_of_opens(opens, OpenType::kPon), 2);
    EXPECT_TRUE(opens[0]->At(0).Is(TileType::kM5));
    EXPECT_TRUE(opens[0]->At(0).IsRedFive());
    EXPECT_TRUE(opens[1]->At(0).Is(TileType::kM5));
    EXPECT_FALSE(opens[1]->At(0).IsRedFive());
    // One possible pon is expected
    h = Hand(HandParams("m1,m1,m1,m2,m4,m4,m4,m6,m7,m8,m9,m9,m9"));
    opens = h.PossibleOpensAfterOthersDiscard(Tile("m4", 3), RelativePos::kMid);
    EXPECT_EQ(num_of_opens(opens, OpenType::kPon), 1);

    // KanOpened
    h = Hand(HandParams("m1,m1,m1,m2,m3,m4,m5,m6,m7,m8,m9,m9,m9"));
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
    h = Hand(HandParams("m2,m3,m4,m4,m4,m5,m5,m6,m7,m8,m9,m9,m9"));
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

TEST(hand, PossibleOpensAfterDraw) {
    // PossibleKanClosed
    auto h = Hand(HandParams("m1,m1,m1,m2,m2,m3,m4,m5,m6,m7,m9,m9,m9"));
    h.Draw(Tile("m9", 3));
    EXPECT_EQ(h.Size(), 14);
    EXPECT_EQ(h.PossibleOpensAfterDraw().size(), 1);
    EXPECT_EQ((*h.PossibleOpensAfterDraw().begin())->Type(), OpenType::kKanClosed);
    EXPECT_EQ((*h.PossibleOpensAfterDraw().begin())->At(0).Type(), TileType::kM9);
    EXPECT_EQ((*h.PossibleOpensAfterDraw().begin())->StolenTile(), Tile("m9", 0));
    EXPECT_EQ((*h.PossibleOpensAfterDraw().begin())->LastTile(), Tile("m9", 0));

    // PossibleKanAdded
    h = Hand(HandParams("m1,m1,m1,m2,m3,m4,m5,m6,m7,m8,m9,m9,m9"));
    h.ApplyOpen(std::make_unique<Pon>(Tile("m9", 3), Tile("m9", 2), RelativePos::kMid));
    h.Discard(Tile("m1", 0));
    h.Draw(Tile("m8", 2));
    EXPECT_EQ(h.Size(), 14);
    EXPECT_EQ(h.SizeClosed(), 11);
    EXPECT_EQ(h.SizeOpened(), 3);
    EXPECT_EQ(h.PossibleOpensAfterDraw().size(), 1);
    EXPECT_EQ((*h.PossibleOpensAfterDraw().begin())->Type(), OpenType::kKanAdded);
    EXPECT_EQ((*h.PossibleOpensAfterDraw().begin())->At(0).Type(), TileType::kM9);
    EXPECT_EQ((*h.PossibleOpensAfterDraw().begin())->StolenTile(), Tile("m9", 3));
    EXPECT_EQ((*h.PossibleOpensAfterDraw().begin())->LastTile(), Tile("m9", 2));

    // mixed
    h = Hand(HandParams("m1,m1,m1,m2,m3,m4,m5,m6,m7,m8,m9,m9,m9"));
    h.ApplyOpen(std::make_unique<Pon>(Tile("m9", 3), Tile("m9", 2), RelativePos::kMid));
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
    auto h = Hand(HandParams("m1,m9,p1,p9,s1,s9,ew,sw,ww,nw,wd,gd,rd"));
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

    auto h = Hand(HandParams("m1,m1,m1,m2,m3,m4,m5,m6,m7,m8,m9,m9,m9"));
    auto chis = h.PossibleOpensAfterOthersDiscard(Tile("m2", 1), RelativePos::kLeft);
    h.ApplyOpen(std::move(chis.at(0)));
    h.Discard(Tile("m9", 2));
    auto pons = h.PossibleOpensAfterOthersDiscard(Tile("m1", 3), RelativePos::kMid);
    h.ApplyOpen(std::move(pons.at(0)));
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

    auto h = Hand(HandParams("m1,m1,m1,m2,m3,m4,m5,m6,m7,m8,m9,m9,m9"));
    auto chis = h.PossibleOpensAfterOthersDiscard(Tile("m2", 1), RelativePos::kLeft);
    h.ApplyOpen(std::move(chis.at(0)));
    h.Discard(Tile("m9", 2));
    auto pons = h.PossibleOpensAfterOthersDiscard(Tile("m1", 3), RelativePos::kMid);
    h.ApplyOpen(std::move(pons.at(0)));
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
    auto h = Hand(HandParams("m1,m1,m1,m2,m3,m4,m5,m6,m7,m8,m9,m9,m9"));
    EXPECT_TRUE(h.IsMenzen());
    h.Draw(Tile("m9", 3));
    auto kans = h.PossibleOpensAfterDraw();
    h.ApplyOpen(std::move(kans.front()));
    h.Draw(Tile("m4", 3));
    EXPECT_TRUE(h.IsMenzen());
    h.Discard(Tile("m4", 0));
    auto chis = h.PossibleOpensAfterOthersDiscard(Tile("m5", 3), RelativePos::kLeft);
    h.ApplyOpen(std::move(chis.front()));
    EXPECT_FALSE(h.IsMenzen());
}

TEST(hand, CanRon) {
    auto h = Hand(HandParams("m1,m1,m1,m2,m3,m4,m5,m6,m7,m8,m9,m9,m9"));
    EXPECT_TRUE(h.CanRon(Tile("m1", 3)));
    EXPECT_TRUE(h.CanRon(Tile("m5", 3)));
    EXPECT_TRUE(h.CanRon(Tile("m9", 3)));
    h = Hand(HandParams("m1,m1,m1,m2,m3,m4,m5,m6,m7,m8,m9,m9,rd"));
    EXPECT_FALSE(h.CanRon(Tile("m1", 3)));
}

TEST(hand, IsCompleted) {
    auto h = Hand(HandParams("m1,m1,m1,m2,m3,m4,m5,m6,m7,m8,m9,m9,m9"));
    h.Draw(Tile("m1", 3));
    EXPECT_TRUE(h.IsCompleted());
    h = Hand(HandParams("m1,m1,m1,m2,m3,m4,m5,m6,m7,m8,m9,m9,m9"));
    h.Draw(Tile("rd", 0));
    EXPECT_FALSE(h.IsCompleted());
}

TEST(hand, CanRiichi) {
    auto h = Hand(HandParams("m1,m1,m1,m2,m3,m4,m5,m6,m7,m8,m9,m9,m9"));
    h.Draw(Tile("p1"));
    EXPECT_TRUE(h.CanRiichi());
    h = Hand(HandParams("m1,m1,m1,m2,m3,m4,m5,m6,m7,m8,m9,m9,p9"));
    h.Draw(Tile("p1"));
    EXPECT_FALSE(h.CanRiichi());
    h = Hand(HandParams("m1,m1,m1,m2,m3,m4,m5,m6,m7,m8,m9,m9,m9"));
    auto chis = h.PossibleOpensAfterOthersDiscard(Tile("m1", 2), RelativePos::kLeft);
    h.ApplyOpen(std::move(chis.at(0)));
    h.Discard(Tile("m9"));
    h.Draw(Tile("p1"));
    EXPECT_FALSE(h.CanRiichi());
}

TEST(hand, Opens) {
    auto h = Hand(HandParams("m1,m1,m2,m3,m4,m5,m6,m7,m8,m9,m9,m9,m9"));
    auto chis = h.PossibleOpensAfterOthersDiscard(Tile("m1", 2), RelativePos::kLeft);
    h.ApplyOpen(std::move(chis.at(0)));
    const auto opens = h.Opens();
    EXPECT_EQ(opens.size(), 1);
    EXPECT_EQ(opens.front()->Type(), OpenType::kChi);
}

TEST(hand, Riichi) {
    auto h = Hand(HandParams("m1,m1,m1,m2,m3,m4,m5,m6,m7,m8,m9,m9,m9"));
    h.Draw(Tile("rd"));
    EXPECT_FALSE(h.IsUnderRiichi());
    h.Riichi();
    EXPECT_TRUE(h.IsUnderRiichi());
}

TEST(hand, PossibleDiscardsAfterRiichi) {
    auto h = Hand(HandParams("m1,m1,m1,m2,m3,m4,m5,m6,m7,m8,m9,m9,m9"));
    h.Draw(Tile("rd"));
    h.Riichi();
    auto possible_discards = h.PossibleDiscardsAfterRiichi();
    EXPECT_EQ(possible_discards.size(), 4);
    auto HasType = [&](TileType tt) {
        return std::find_if(possible_discards.begin(), possible_discards.end(),
                             [&](auto x){ return x.Type() == tt; })!= possible_discards.end();
    };
    EXPECT_TRUE(HasType(TileType::kRD));
    EXPECT_TRUE(HasType(TileType::kM2));
    EXPECT_TRUE(HasType(TileType::kM5));
    EXPECT_TRUE(HasType(TileType::kM8));
}

TEST(hand, ToString) {
    auto h = Hand(HandParams("m1,m1,m1,m2,m3,m4,m5,m6,m7,m8,m9,m9,m9"));
    EXPECT_EQ(h.ToString(), "m1,m1,m1,m2,m3,m4,m5,m6,m7,m8,m9,m9,m9");
    EXPECT_EQ(h.ToString(true), "m1(0),m1(1),m1(2),m2(0),m3(0),m4(0),m5(0),m6(0),m7(0),m8(0),m9(0),m9(1),m9(2)");
    auto possible_opens = h.PossibleOpensAfterOthersDiscard(Tile("m5", 3), RelativePos::kLeft);
    h.ApplyOpen(std::move(possible_opens.front()));
    EXPECT_EQ(h.ToString(), "m1,m1,m1,m2,m3,m4,m5,m8,m9,m9,m9,[m5,m6,m7]");
    EXPECT_EQ(h.ToString(true), "m1(0),m1(1),m1(2),m2(0),m3(0),m4(0),m5(0),m8(0),m9(0),m9(1),m9(2),[m5(3),m6(0),m7(0)]");
    h.Discard(Tile("m1", 0));
    h.Draw(Tile("m9", 3));
    possible_opens = h.PossibleOpensAfterDraw();
    h.ApplyOpen(std::move(possible_opens.front()));
    EXPECT_EQ(h.ToString(), "m1,m1,m2,m3,m4,m5,m8,[m5,m6,m7],[m9,m9,m9,m9]c");
    EXPECT_EQ(h.ToString(true), "m1(1),m1(2),m2(0),m3(0),m4(0),m5(0),m8(0),[m5(3),m6(0),m7(0)],[m9(0),m9(1),m9(2),m9(3)]c");
}

TEST(hand, LastTileAdded) {
    auto h = Hand(HandParams("m1,m1,m1,m2,m3,m4,m5,m6,m7,m8,m9,m9,m9"));
    EXPECT_TRUE(h.LastTileAdded() == std::nullopt);
    h.Draw(Tile("m1", 3));
    EXPECT_EQ(h.LastTileAdded(), Tile("m1", 3));
    h.Discard(Tile("m1", 0));
    EXPECT_TRUE(h.LastTileAdded() == std::nullopt);
    auto opens = h.PossibleOpensAfterOthersDiscard(Tile("m2", 3), RelativePos::kLeft);
    h.ApplyOpen(std::move(opens.front()));
    EXPECT_TRUE(h.LastTileAdded() == Tile("m2", 3));
    h.Discard(Tile("m1", 1));
    EXPECT_TRUE(h.LastTileAdded() == std::nullopt);
}

TEST(hand, Ron) {
    auto h = Hand(HandParams("m1,m1,m1,m2,m3,m4,m5,m6,m7,m8,m9,m9,m9"));
    h.Ron(Tile("m1", 3));
    EXPECT_EQ(h.Stage(), HandStage::kAfterRon);
    EXPECT_EQ(h.LastTileAdded(), Tile("m1", 3));
}

TEST(hand, Tsumo) {
    auto h = Hand(HandParams("m1,m1,m1,m2,m3,m4,m5,m6,m7,m8,m9,m9,m9"));
    h.Draw(Tile("m1", 3));
    h.Tsumo();
    EXPECT_EQ(h.Stage(), HandStage::kAfterTsumo);
    EXPECT_EQ(h.LastTileAdded(), Tile("m1", 3));

    // after kan
    h = Hand(HandParams("m1,m1,m1,m2,m3,m4,m5,m6,m7,m8,m9,m9,m9"));
    h.Draw(Tile("m9", 3));
    auto possible_opens = h.PossibleOpensAfterDraw();
    h.ApplyOpen(std::move(possible_opens.front()));
    h.Draw(Tile("m1", 3));
    h.Tsumo();
    EXPECT_EQ(h.Stage(), HandStage::kAfterTsumoAfterKan);
    EXPECT_EQ(h.LastTileAdded(), Tile("m1", 3));
}

TEST(hand, RonAfterOhtersKan) {
    auto h = Hand(HandParams("m1,m1,m1,m2,m3,m4,m5,m6,m7,m8,m9,m9,m9"));
    h.RonAfterOthersKan(Tile("m1", 3));
    EXPECT_EQ(h.Stage(), HandStage::kAfterRonAfterOthersKan);
    EXPECT_EQ(h.LastTileAdded(), Tile("m1", 3));
}

TEST(hand, EvalScore) {
    auto h = Hand(HandParams("m1,m1,m1,m2,m3,m4,s3,s3,p2,p2,sw,sw,sw").Tsumo("p2"));

    auto score = h.EvalScore(WinningStateInfo().PrevalentWind(Wind::kSouth).IsBottom(true)
            .IsFirstTsumo(false).Dora({TileType::kM1}));

    EXPECT_EQ(score.HasYaku(Yaku::kPrevalentWindSouth), std::make_optional(1));
    EXPECT_EQ(score.HasYaku(Yaku::kBottomOfTheSea), std::make_optional(1));
    EXPECT_EQ(score.HasYaku(Yaku::kDora), std::make_optional(3));
}