#include <mjx/internal/hand.h>
#include <mjx/internal/tile.h>

#include <array>
#include <optional>

#include "gtest/gtest.h"

using namespace mjx::internal;

TEST(internal_hand, Hand) {
  EXPECT_NO_FATAL_FAILURE(
      Hand(Tile::Create({"m1", "m9", "p1", "p9", "s1", "s9", "ew", "sw", "ww",
                         "nw", "wd", "gd", "rd"})));
  auto tiles = Tile::Create({"m1", "m9", "p1", "p9", "s1", "s9", "ew", "sw",
                             "ww", "nw", "wd", "gd", "rd"});
  EXPECT_NO_FATAL_FAILURE(Hand(tiles.begin(), tiles.end()));

  auto hand = Hand(HandParams("m1,m2,m3,m4,m5,rd,rd")
                       .Chi("m7,m8,m9")
                       .KanAdded("p1,p1,p1,p1"));
  auto actual = hand.ToVector(true);
  auto expected = Tile::Create({"m1", "m2", "m3", "m4", "m5", "rd", "rd", "m7",
                                "m8", "m9", "p1", "p1", "p1", "p1"},
                               true);
  EXPECT_EQ(actual, expected);
  actual = hand.ToVectorClosed(true);
  expected = Tile::Create({"m1", "m2", "m3", "m4", "m5", "rd", "rd"}, true);
  EXPECT_EQ(actual, expected);
  actual = hand.ToVectorOpened(true);
  expected = Tile::Create({"m7", "m8", "m9", "p1", "p1", "p1", "p1"}, true);
  EXPECT_EQ(actual, expected);

  hand = Hand(HandParams("m1,m2,m3,m4,m5,wd,wd,wd,rd,rd").Chi("m7,m8,m9"));
  actual = hand.ToVector(true);
  expected = Tile::Create({{"m1", "m2", "m3", "m4", "m5", "rd", "rd", "wd",
                            "wd", "wd", "m7", "m8", "m9"}},
                          true);
  EXPECT_EQ(actual, expected);
  actual = hand.ToVectorClosed(true);
  expected = Tile::Create(
      {"m1", "m2", "m3", "m4", "m5", "rd", "rd", "wd", "wd", "wd"}, true);
  EXPECT_EQ(actual, expected);
  actual = hand.ToVectorOpened(true);
  expected = Tile::Create({{"m7", "m8", "m9"}}, true);
  EXPECT_EQ(actual, expected);

  hand =
      Hand(HandParams("m4,m5,rd,rd,wd,wd,wd").Chi("m1,m2,m3").Chi("m7,m8,m9"));
  actual = hand.ToVector(true);
  expected = Tile::Create({"m4", "m5", "rd", "rd", "wd", "wd", "wd", "m1", "m2",
                           "m3", "m7", "m8", "m9"},
                          true);
  EXPECT_EQ(actual, expected);
  actual = hand.ToVectorClosed(true);
  expected = Tile::Create({"m4", "m5", "rd", "rd", "wd", "wd", "wd"}, true);
  EXPECT_EQ(actual, expected);
  actual = hand.ToVectorOpened(true);
  expected = Tile::Create({"m1", "m2", "m3", "m7", "m8", "m9"}, true);
  EXPECT_EQ(actual, expected);

  hand = Hand(HandParams("m1,m2,m3,m4,m5,rd,rd,wd,wd,wd").Pon("p3,p3,p3"));
  actual = hand.ToVector(true);
  expected = Tile::Create({"m1", "m2", "m3", "m4", "m5", "rd", "rd", "wd", "wd",
                           "wd", "p3", "p3", "p3"},
                          true);
  EXPECT_EQ(actual, expected);
  actual = hand.ToVectorClosed(true);
  expected = Tile::Create(
      {"m1", "m2", "m3", "m4", "m5", "rd", "rd", "wd", "wd", "wd"}, true);
  EXPECT_EQ(actual, expected);
  actual = hand.ToVectorOpened(true);
  expected = Tile::Create({"p3", "p3", "p3"}, true);
  EXPECT_EQ(actual, expected);

  hand =
      Hand(HandParams("m1,m2,m3,m4,m5,rd,rd").Pon("p3,p3,p3").Pon("wd,wd,wd"));
  actual = hand.ToVector(true);
  expected = Tile::Create({"m1", "m2", "m3", "m4", "m5", "rd", "rd", "p3", "p3",
                           "p3", "wd", "wd", "wd"},
                          true);
  EXPECT_EQ(actual, expected);
  actual = hand.ToVectorClosed(true);
  expected = Tile::Create({"m1", "m2", "m3", "m4", "m5", "rd", "rd"}, true);
  EXPECT_EQ(actual, expected);
  actual = hand.ToVectorOpened(true);
  expected = Tile::Create({"p3", "p3", "p3", "wd", "wd", "wd"}, true);
  EXPECT_EQ(actual, expected);

  hand = Hand(HandParams("nw")
                  .KanOpened("p3,p3,p3,p3")
                  .KanOpened("wd,wd,wd,wd")
                  .KanOpened("rd,rd,rd,rd")
                  .KanOpened("gd,gd,gd,gd"));
  actual = hand.ToVector(true);
  expected = Tile::Create({"nw", "p3", "p3", "p3", "p3", "wd", "wd", "wd", "wd",
                           "rd", "rd", "rd", "rd", "gd", "gd", "gd", "gd"},
                          true);
  EXPECT_EQ(actual, expected);
  actual = hand.ToVectorClosed(true);
  expected = Tile::Create(std::vector<std::string>({"nw"}), true);
  EXPECT_EQ(actual, expected);
  actual = hand.ToVectorOpened(true);
  expected = Tile::Create({"p3", "p3", "p3", "p3", "wd", "wd", "wd", "wd", "rd",
                           "rd", "rd", "rd", "gd", "gd", "gd", "gd"},
                          true);
  EXPECT_EQ(actual, expected);

  hand = Hand(HandParams("nw")
                  .KanClosed("p3,p3,p3,p3")
                  .KanClosed("wd,wd,wd,wd")
                  .KanClosed("rd,rd,rd,rd")
                  .KanClosed("gd,gd,gd,gd"));
  actual = hand.ToVector(true);
  expected = Tile::Create({"nw", "p3", "p3", "p3", "p3", "wd", "wd", "wd", "wd",
                           "rd", "rd", "rd", "rd", "gd", "gd", "gd", "gd"},
                          true);
  EXPECT_EQ(actual, expected);
  actual = hand.ToVectorClosed(true);
  expected = Tile::Create(std::vector<std::string>({"nw"}), true);
  EXPECT_EQ(actual, expected);
  actual = hand.ToVectorOpened(true);
  expected = Tile::Create({"p3", "p3", "p3", "p3", "wd", "wd", "wd", "wd", "rd",
                           "rd", "rd", "rd", "gd", "gd", "gd", "gd"},
                          true);
  EXPECT_EQ(actual, expected);

  hand = Hand(HandParams("nw")
                  .KanAdded("p3,p3,p3,p3")
                  .KanAdded("wd,wd,wd,wd")
                  .KanAdded("rd,rd,rd,rd")
                  .KanAdded("gd,gd,gd,gd"));
  actual = hand.ToVector(true);
  expected = Tile::Create({"nw", "p3", "p3", "p3", "p3", "wd", "wd", "wd", "wd",
                           "rd", "rd", "rd", "rd", "gd", "gd", "gd", "gd"},
                          true);
  EXPECT_EQ(actual, expected);
  actual = hand.ToVectorClosed(true);
  expected = Tile::Create(std::vector<std::string>({"nw"}), true);
  EXPECT_EQ(actual, expected);
  actual = hand.ToVectorOpened(true);
  expected = Tile::Create({"p3", "p3", "p3", "p3", "wd", "wd", "wd", "wd", "rd",
                           "rd", "rd", "rd", "gd", "gd", "gd", "gd"},
                          true);
  EXPECT_EQ(actual, expected);

  hand = Hand(HandParams("m1,m2,m3,m4,m5,rd,rd,m7,m8,m9")
                  .KanClosed("p1,p1,p1,p1")
                  .Riichi()
                  .Tsumo("m6"));
  EXPECT_TRUE(hand.IsMenzen());
  EXPECT_TRUE(hand.IsUnderRiichi());
}

TEST(internal_hand, Draw) {
  auto h = Hand(HandParams("m1,m9,p1,p9,s1,s9,ew,sw,ww,nw,wd,gd,rd"));
  EXPECT_EQ(h.Size(), 13);
  h.Draw(Tile(1));
  EXPECT_EQ(h.stage(), HandStage::kAfterDraw);
  EXPECT_EQ(h.Size(), 14);
}

TEST(internal_hand, ApplyChi) {
  auto h = Hand(HandParams("m1,m1,m1,m2,m3,m4,m5,m6,m7,m8,m9,m9,m9"));
  std::vector<Tile> t = {Tile("m2"), Tile("m3"), Tile("m4", 3)};
  auto c = Chi::Create(t, Tile("m4", 3));
  EXPECT_EQ(h.stage(), HandStage::kAfterDiscards);
  EXPECT_EQ(h.Size(), 13);
  h.ApplyOpen(std::move(c));
  EXPECT_EQ(h.stage(), HandStage::kAfterChi);
  EXPECT_EQ(h.Size(), 14);
  EXPECT_EQ(h.SizeOpened(), 3);
  EXPECT_EQ(h.SizeClosed(), 11);
  auto possible_discards = h.PossibleDiscards();
  EXPECT_EQ(possible_discards.size(), 5);  // m5, m6, m7, m8, m9
  EXPECT_EQ(
      std::find_if(possible_discards.begin(), possible_discards.end(),
                   [](const auto& x) { return x.first.Is(TileType::kM4); }),
      possible_discards.end());
  EXPECT_EQ(
      std::find_if(possible_discards.begin(), possible_discards.end(),
                   [](const auto& x) { return x.first.Is(TileType::kM1); }),
      possible_discards.end());
  EXPECT_NE(
      std::find_if(possible_discards.begin(), possible_discards.end(),
                   [](const auto& x) { return x.first.Is(TileType::kM5); }),
      possible_discards.end());
}

TEST(internal_hand, ApplyPon) {
  auto h = Hand(HandParams("m1,m1,m1,m2,m3,m4,m5,m6,m7,m8,m9,m9,m9"));
  auto p = Pon::Create(Tile("m9", 3), Tile("m9", 0), RelativePos::kLeft);
  EXPECT_EQ(h.stage(), HandStage::kAfterDiscards);
  EXPECT_EQ(h.Size(), 13);
  h.ApplyOpen(std::move(p));
  EXPECT_EQ(h.stage(), HandStage::kAfterPon);
  EXPECT_EQ(h.Size(), 14);
  EXPECT_EQ(h.SizeOpened(), 3);
  EXPECT_EQ(h.SizeClosed(), 11);
  auto possible_discards = h.PossibleDiscards();
  EXPECT_EQ(possible_discards.size(), 8);  // m1, m2, m3, m4, m5, m6, m7, m8
  EXPECT_EQ(
      std::find_if(possible_discards.begin(), possible_discards.end(),
                   [](const auto& x) { return x.first.Is(TileType::kM9); }),
      possible_discards.end());
  EXPECT_NE(
      std::find_if(possible_discards.begin(), possible_discards.end(),
                   [](const auto& x) { return x.first.Is(TileType::kM5); }),
      possible_discards.end());
}

TEST(internal_hand, ApplyKanOpened) {
  auto h = Hand(HandParams("m1,m1,m1,m2,m3,m4,m5,m6,m7,m8,m9,m9,m9"));
  auto k = KanOpened::Create(Tile("m9", 3), RelativePos::kMid);
  EXPECT_EQ(h.stage(), HandStage::kAfterDiscards);
  EXPECT_EQ(h.Size(), 13);
  h.ApplyOpen(std::move(k));
  EXPECT_EQ(h.stage(), HandStage::kAfterKanOpened);
  EXPECT_EQ(h.Size(), 14);
  EXPECT_EQ(h.SizeOpened(), 4);
  EXPECT_EQ(h.SizeClosed(), 10);
  h.Draw(Tile("m3", 3));
  auto possible_discards = h.PossibleDiscards();
  EXPECT_EQ(possible_discards.size(),
            9);  // m1, m2, m3, m3(draw), m4, m5, m6, m7, m8
}

TEST(internal_hand, ApplyKanClosed) {
  auto h = Hand(HandParams("m1,m1,m1,m2,m3,m4,m5,m6,m7,m8,m9,m9,m9"));
  h.Draw(Tile("m9", 3));
  auto k = KanClosed::Create(Tile("m9", 0));
  EXPECT_EQ(h.stage(), HandStage::kAfterDraw);
  EXPECT_EQ(h.Size(), 14);
  h.ApplyOpen(std::move(k));
  EXPECT_EQ(h.stage(), HandStage::kAfterKanClosed);
  EXPECT_EQ(h.Size(), 14);
  EXPECT_EQ(h.SizeOpened(), 4);
  EXPECT_EQ(h.SizeClosed(), 10);
  h.Draw(Tile("m3", 3));
  auto possible_discards = h.PossibleDiscards();
  EXPECT_EQ(possible_discards.size(),
            9);  // m1, m2, m3, m3(draw), m4, m5, m6, m7, m8
}

TEST(internal_hand, ApplyKanAdded) {
  auto h = Hand(HandParams("m1,m1,m1,m2,m3,m4,m5,m6,m7,m8,m8,m9,m9"));
  auto p = Pon::Create(Tile("m9", 2), Tile("m9", 3), RelativePos::kLeft);
  auto k = KanAdded::Create(p);
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
  EXPECT_EQ(h.stage(), HandStage::kAfterKanAdded);
}

TEST(internal_hand, Discard) {
  auto h = Hand(HandParams("m1,m1,p1,p2,p3,s9,ew,sw,ww,nw,wd,gd,rd"));
  EXPECT_EQ(h.Size(), 13);
  h.Draw(Tile("rd", 2));
  EXPECT_EQ(h.Size(), 14);
  EXPECT_EQ(h.stage(), HandStage::kAfterDraw);
  h.Discard(Tile("rd"));
  EXPECT_EQ(h.Size(), 13);
  EXPECT_EQ(h.stage(), HandStage::kAfterDiscards);

  // Tsumogiri
  h = Hand(HandParams("m1,m1,p1,p2,p3,s9,ew,sw,ww,nw,wd,gd,rd"));
  auto draw = Tile("p5");
  h.Draw(draw);
  auto [discarded1, tsumogiri1] = h.Discard(draw);
  EXPECT_TRUE(tsumogiri1);
  h = Hand(HandParams("m1,m1,p1,p2,p3,s9,ew,sw,ww,nw,wd,gd,rd"));
  draw = Tile("p5");
  h.Draw(draw);
  auto [discarded2, tsumogiri2] = h.Discard(Tile("m1", 0));
  EXPECT_FALSE(tsumogiri2);
}

TEST(internal_hand, PossibleDiscards) {
  auto h = Hand(HandParams("m1,m2,m3,p9,s1,s9,ew,sw,ww,nw,wd,gd,rd"));
  auto t = Tile::Create({"m1", "m2", "m3"});
  auto c = Chi::Create(t, Tile("m3", 0));
  h.ApplyOpen(std::move(c));
  auto possible_discards = h.PossibleDiscards();
  EXPECT_EQ(possible_discards.size(), 10);
  EXPECT_EQ(
      std::find_if(possible_discards.begin(), possible_discards.end(),
                   [](const auto& x) { return x.first.Is(TileType::kM3); }),
      possible_discards.end());
}

TEST(internal_hand, PossibleDiscardsToTakeTenpai) {
  auto h = Hand(HandParams("m1,m2,m3,s1,s2,s3,s4,s5,s6,s1,s2,ew,nw"));
  h.Draw(Tile("ew", 2));
  auto possible_discards = h.PossibleDiscardsToTakeTenpai();
  EXPECT_EQ(possible_discards.size(), 1);
  EXPECT_EQ(possible_discards.front().first.Type(), TileType::kNW);

  // From actual failure
  h = Hand({Tile("m2", 2), Tile("m3", 1), Tile("m3", 3), Tile("m4", 2),
            Tile("m7", 2), Tile("m7", 3), Tile("s3", 3), Tile("s4", 2),
            Tile("rd", 0), Tile("rd", 1), Tile("rd", 2), Tile("wd", 0),
            Tile("wd", 1)});
  h.Draw(Tile("wd", 2));
  possible_discards = h.PossibleDiscardsToTakeTenpai();
  EXPECT_EQ(possible_discards.size(), 1);
  EXPECT_EQ(possible_discards.front().first.Id(), Tile("m3", 1).Id());
}

TEST(internal_hand,
     PossibleOpensAfterOthersDiscard) {  // TODO: add more detailed test
  auto num_of_opens = [](const auto& opens, const auto& open_type) {
    return std::count_if(
        opens.begin(), opens.end(),
        [&open_type](const auto& x) { return x.Type() == open_type; });
  };

  // Chi
  auto h = Hand(HandParams("m1,m1,m1,m2,m3,m4,m5,m6,m7,m8,m9,m9,m9"));
  // [m1]m2m3
  auto opens =
      h.PossibleOpensAfterOthersDiscard(Tile("m1", 3), RelativePos::kLeft);
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
  // m4m5m6m7[m7] Kuikae: no chi is expected
  h = Hand(HandParams("m4,m5,m6,m7")
               .Chi("s1,s2,s3")
               .Chi("s7,s8,s9")
               .Pon("p3,p3,p3"));
  opens = h.PossibleOpensAfterOthersDiscard(Tile("m7", 3), RelativePos::kLeft);
  EXPECT_EQ(num_of_opens(opens, OpenType::kChi), 0);

  // Pon
  // No pon is expected
  h = Hand(HandParams("m1,m1,m1,m2,m3,m4,m5,m6,m7,m8,m9,m9,m9"));
  opens = h.PossibleOpensAfterOthersDiscard(Tile("m5", 3), RelativePos::kMid);
  EXPECT_EQ(num_of_opens(opens, OpenType::kPon), 0);
  // One possible pon is expected
  h = Hand(HandParams("m1,m1,m1,m2,m3,m5,m5,m6,m7,m8,m9,m9,m9"));
  opens = h.PossibleOpensAfterOthersDiscard(Tile("m5", 3), RelativePos::kMid);
  EXPECT_EQ(num_of_opens(opens, OpenType::kPon), 1);
  EXPECT_EQ(opens[0].Type(), OpenType::kPon);
  EXPECT_EQ(opens[0].At(0).Type(), TileType::kM5);
  // Two possible pons are expected (w/ red 5 and w/o red 5)
  h = Hand(HandParams("m1,m1,m1,m2,m5,m5,m5,m6,m7,m8,m9,m9,m9"));
  opens = h.PossibleOpensAfterOthersDiscard(Tile("m5", 3), RelativePos::kMid);
  EXPECT_EQ(num_of_opens(opens, OpenType::kPon), 2);
  EXPECT_TRUE(opens[0].At(0).Is(TileType::kM5));
  EXPECT_TRUE(opens[0].At(0).IsRedFive());
  EXPECT_TRUE(opens[1].At(0).Is(TileType::kM5));
  EXPECT_FALSE(opens[1].At(0).IsRedFive());
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
  EXPECT_EQ(opens.back().Type(), OpenType::kKanOpened);
  EXPECT_EQ(opens.back().At(0).Type(), TileType::kM1);
  EXPECT_EQ(opens.back().StolenTile(), Tile("m1", 3));
  EXPECT_EQ(opens.back().LastTile(), Tile("m1", 3));
  opens = h.PossibleOpensAfterOthersDiscard(Tile("m2", 3), RelativePos::kMid);
  EXPECT_EQ(num_of_opens(opens, OpenType::kKanOpened), 0);
  opens = h.PossibleOpensAfterOthersDiscard(Tile("m9", 3), RelativePos::kMid);
  EXPECT_EQ(opens.size(), 2);
  EXPECT_EQ(num_of_opens(opens, OpenType::kKanOpened), 1);
  EXPECT_EQ(opens.back().Type(), OpenType::kKanOpened);
  EXPECT_EQ(opens.back().At(0).Type(), TileType::kM9);
  EXPECT_EQ(opens.back().StolenTile(), Tile("m9", 3));
  EXPECT_EQ(opens.back().LastTile(), Tile("m9", 3));

  // Mixed
  h = Hand(HandParams("m2,m3,m4,m4,m4,m5,m5,m6,m7,m8,m9,m9,m9"));
  auto possible_opens =
      h.PossibleOpensAfterOthersDiscard(Tile("m4", 3), RelativePos::kLeft);
  // chi [m4]m5m6, [m4]*m5m6, m3[m4]m5, m3[m4]*m5, m2m3[m4]
  // pon m4m4m4
  // kan m4m4m4m4
  EXPECT_EQ(possible_opens.size(), 7);
  EXPECT_EQ(possible_opens.at(0).Type(), OpenType::kChi);
  EXPECT_EQ(possible_opens.at(0).At(0).Type(), TileType::kM4);
  EXPECT_TRUE(possible_opens.at(0).At(1).IsRedFive());
  EXPECT_EQ(possible_opens.at(1).Type(), OpenType::kChi);
  EXPECT_EQ(possible_opens.at(1).At(0).Type(), TileType::kM4);
  EXPECT_TRUE(!possible_opens.at(1).At(1).IsRedFive());
  EXPECT_EQ(possible_opens.at(2).Type(), OpenType::kChi);
  EXPECT_EQ(possible_opens.at(2).At(0).Type(), TileType::kM3);
  EXPECT_TRUE(possible_opens.at(2).At(2).IsRedFive());
  EXPECT_EQ(possible_opens.at(3).Type(), OpenType::kChi);
  EXPECT_EQ(possible_opens.at(3).At(0).Type(), TileType::kM3);
  EXPECT_TRUE(!possible_opens.at(3).At(2).IsRedFive());
  EXPECT_EQ(possible_opens.at(4).Type(), OpenType::kChi);
  EXPECT_EQ(possible_opens.at(4).At(0).Type(), TileType::kM2);
  EXPECT_EQ(possible_opens.at(5).Type(), OpenType::kPon);
  EXPECT_EQ(possible_opens.at(6).Type(), OpenType::kKanOpened);
}

TEST(internal_hand, PossibleOpensAfterDraw) {
  // PossibleKanClosed
  auto h = Hand(HandParams("m1,m1,m1,m2,m2,m3,m4,m5,m6,m7,m9,m9,m9"));
  h.Draw(Tile("m9", 3));
  EXPECT_EQ(h.Size(), 14);
  EXPECT_EQ(h.PossibleOpensAfterDraw().size(), 1);
  EXPECT_EQ((*h.PossibleOpensAfterDraw().begin()).Type(), OpenType::kKanClosed);
  EXPECT_EQ((*h.PossibleOpensAfterDraw().begin()).At(0).Type(), TileType::kM9);
  EXPECT_EQ((*h.PossibleOpensAfterDraw().begin()).StolenTile(), Tile("m9", 0));
  EXPECT_EQ((*h.PossibleOpensAfterDraw().begin()).LastTile(), Tile("m9", 0));

  // PossibleKanAdded
  h = Hand(HandParams("m1,m1,m1,m2,m3,m4,m5,m6,m7,m8,m9,m9,m9"));
  h.ApplyOpen(Pon::Create(Tile("m9", 3), Tile("m9", 2), RelativePos::kMid));
  h.Discard(Tile("m1", 0));
  h.Draw(Tile("m8", 2));
  EXPECT_EQ(h.Size(), 14);
  EXPECT_EQ(h.SizeClosed(), 11);
  EXPECT_EQ(h.SizeOpened(), 3);
  EXPECT_EQ(h.PossibleOpensAfterDraw().size(), 1);
  EXPECT_EQ((*h.PossibleOpensAfterDraw().begin()).Type(), OpenType::kKanAdded);
  EXPECT_EQ((*h.PossibleOpensAfterDraw().begin()).At(0).Type(), TileType::kM9);
  EXPECT_EQ((*h.PossibleOpensAfterDraw().begin()).StolenTile(), Tile("m9", 3));
  EXPECT_EQ((*h.PossibleOpensAfterDraw().begin()).LastTile(), Tile("m9", 2));

  // mixed
  h = Hand(HandParams("m1,m1,m1,m2,m3,m4,m5,m6,m7,m8,m9,m9,m9"));
  h.ApplyOpen(Pon::Create(Tile("m9", 3), Tile("m9", 2), RelativePos::kMid));
  h.Discard(Tile("m3", 0));
  h.Draw(Tile("m1", 3));
  auto possible_opens = h.PossibleOpensAfterDraw();
  EXPECT_EQ(possible_opens.size(), 2);
  EXPECT_EQ(possible_opens.at(0).Type(), OpenType::kKanClosed);
  EXPECT_EQ(possible_opens.at(0).At(0).Type(), TileType::kM1);
  EXPECT_EQ(possible_opens.at(1).Type(), OpenType::kKanAdded);
  EXPECT_EQ(possible_opens.at(1).At(0).Type(), TileType::kM9);

  // リーチ後のカンで待ちが変わるときにはカンできない
  // リーチ後だけどカンできる場合
  h = Hand(HandParams("s3,s3,s3,s5,s6,s6,s6,wd,wd,wd,rd,rd,rd").Riichi());
  h.Draw(Tile("s3", 3));
  possible_opens = h.PossibleOpensAfterDraw();
  EXPECT_EQ(possible_opens.size(), 1);
  EXPECT_EQ(possible_opens.front().Type(), OpenType::kKanClosed);
  EXPECT_EQ(possible_opens.front().At(0).Type(), TileType::kS3);
  // リーチ後でカンできない場合
  h = Hand(HandParams("s3,s3,s3,s5,s6,s6,s6,wd,wd,wd,rd,rd,rd").Riichi());
  h.Draw(Tile("s6", 3));
  possible_opens = h.PossibleOpensAfterDraw();
  EXPECT_EQ(possible_opens.size(), 0);
}

TEST(internal_hand, Size) {
  auto h = Hand(HandParams("m1,m9,p1,p9,s1,s9,ew,sw,ww,nw,wd,gd,rd"));
  EXPECT_EQ(h.Size(), 13);
  EXPECT_EQ(h.SizeClosed(), 13);
  EXPECT_EQ(h.SizeOpened(), 0);
  // TODO : add test cases with melds
}

TEST(internal_hand, ToVector) {
  auto check_vec = [](const std::vector<Tile>& v1,
                      const std::vector<Tile>& v2) {
    for (std::size_t i = 0; i < v1.size(); ++i)
      if (v1.at(i).Type() != v2.at(i).Type()) return false;
    return true;
  };

  auto h = Hand(HandParams("m1,m1,m1,m2,m3,m4,m5,m6,m7,m8,m9,m9,m9"));
  auto chis =
      h.PossibleOpensAfterOthersDiscard(Tile("m2", 1), RelativePos::kLeft);
  h.ApplyOpen(std::move(chis.at(0)));
  h.Discard(Tile("m9", 2));
  auto pons =
      h.PossibleOpensAfterOthersDiscard(Tile("m1", 3), RelativePos::kMid);
  h.ApplyOpen(std::move(pons.at(0)));
  h.Discard(Tile("m9", 1));
  EXPECT_EQ(h.Size(), 13);
  EXPECT_EQ(h.ToVector().size(), 13);
  EXPECT_EQ(h.SizeClosed(), 7);
  EXPECT_EQ(h.ToVectorClosed().size(), 7);
  EXPECT_EQ(h.SizeOpened(), 6);
  EXPECT_EQ(h.ToVectorOpened().size(), 6);
  EXPECT_TRUE(check_vec(h.ToVector(true),
                        Tile::Create({"m1", "m1", "m1", "m1", "m2", "m2", "m3",
                                      "m4", "m5", "m6", "m7", "m8", "m9"})));
  EXPECT_TRUE(
      check_vec(h.ToVectorClosed(true),
                Tile::Create({"m1", "m2", "m5", "m6", "m7", "m8", "m9"})));
  EXPECT_TRUE(check_vec(h.ToVectorOpened(true),
                        Tile::Create({"m1", "m1", "m1", "m2", "m3", "m4"})));
}

TEST(internal_hand, ToArray) {
  auto check_arr = [](const std::array<std::uint8_t, 34>& a1,
                      const std::array<std::uint8_t, 34>& a2) {
    for (std::size_t i = 0; i < 34; ++i) {
      if (a1.at(i) != a2.at(i)) return false;
    }
    return true;
  };

  auto h = Hand(HandParams("m1,m1,m1,m2,m3,m4,m5,m6,m7,m8,m9,m9,m9"));
  auto chis =
      h.PossibleOpensAfterOthersDiscard(Tile("m2", 1), RelativePos::kLeft);
  h.ApplyOpen(std::move(chis.at(0)));
  h.Discard(Tile("m9", 2));
  auto pons =
      h.PossibleOpensAfterOthersDiscard(Tile("m1", 3), RelativePos::kMid);
  h.ApplyOpen(std::move(pons.at(0)));
  h.Discard(Tile("m9", 1));
  std::array<std::uint8_t, 34> expected = {4, 2, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
                                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  EXPECT_TRUE(check_arr(h.ToArray(), expected));
  expected = {1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  EXPECT_TRUE(check_arr(h.ToArrayClosed(), expected));
  expected = {3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  EXPECT_TRUE(check_arr(h.ToArrayOpened(), expected));
}

TEST(internal_hand, IsMenzen) {
  // menzen
  auto h = Hand(HandParams("m1,m1,m1,m2,m3,m4,m5,m6,m7,m8,m9,m9,m9"));
  EXPECT_TRUE(h.IsMenzen());
  h.Draw(Tile("m9", 3));
  auto kans = h.PossibleOpensAfterDraw();
  h.ApplyOpen(std::move(kans.front()));
  h.Draw(Tile("m4", 3));
  EXPECT_TRUE(h.IsMenzen());
  h.Discard(Tile("m4", 0));
  auto chis =
      h.PossibleOpensAfterOthersDiscard(Tile("m5", 3), RelativePos::kLeft);
  h.ApplyOpen(std::move(chis.front()));
  EXPECT_FALSE(h.IsMenzen());
}

// TEST(internal_hand, CanRon) {
//     auto h = Hand(HandParams("m1,m1,m1,m2,m3,m4,m5,m6,m7,m8,m9,m9,m9"));
//     EXPECT_TRUE(h.CanRon(Tile("m1", 3)));
//     EXPECT_TRUE(h.CanRon(Tile("m5", 3)));
//     EXPECT_TRUE(h.CanRon(Tile("m9", 3)));
//     h = Hand(HandParams("m1,m1,m1,m2,m3,m4,m5,m6,m7,m8,m9,m9,rd"));
//     EXPECT_FALSE(h.CanRon(Tile("m1", 3)));
// }

TEST(internal_hand, IsCompleted) {
  auto h = Hand(HandParams("m1,m1,m1,m2,m3,m4,m5,m6,m7,m8,m9,m9,m9"));
  h.Draw(Tile("m1", 3));
  EXPECT_TRUE(h.IsCompleted());
  h = Hand(HandParams("m1,m1,m1,m2,m3,m4,m5,m6,m7,m8,m9,m9,m9"));
  h.Draw(Tile("rd", 0));
  EXPECT_FALSE(h.IsCompleted());

  // with additional tile
  h = Hand(HandParams("m1,m1,m1,m2,m3,m4,m5,m6,m7,m8,m9,m9,m9"));
  EXPECT_TRUE(h.IsCompleted(Tile("m1", 3)));
  h = Hand(HandParams("m1,m1,m1,m2,m3,m4,m5,m6,m7,m8,m9,m9,m9"));
  EXPECT_FALSE(h.IsCompleted(Tile("rd", 0)));
}

TEST(internal_hand, CanRiichi) {
  auto h = Hand(HandParams("m1,m1,m1,m2,m3,m4,m5,m6,m7,m8,m9,m9,m9"));
  h.Draw(Tile("p1"));
  EXPECT_TRUE(h.CanRiichi());
  h = Hand(HandParams("m1,m1,m1,m2,m3,m4,m5,m6,m7,m8,m9,m9,p9"));
  h.Draw(Tile("p1"));
  EXPECT_FALSE(h.CanRiichi());
  h = Hand(HandParams("m1,m1,m1,m2,m3,m4,m5,m6,m7,m8,m9,m9,m9"));
  auto chis =
      h.PossibleOpensAfterOthersDiscard(Tile("m1", 2), RelativePos::kLeft);
  h.ApplyOpen(std::move(chis.at(0)));
  h.Discard(Tile("m9"));
  h.Draw(Tile("p1"));
  EXPECT_FALSE(h.CanRiichi());
  // 国士無双
  h = Hand(HandParams("m1,m2,m9,p1,p1,p9,s1,s9,ew,sw,ww,nw,wd"));
  h.Draw(Tile("rd"));
  EXPECT_TRUE(h.CanRiichi());
  // 九種九牌だが国士無双シャンテンではない（ので立直できない）
  h = Hand(
      Tile::Create({2, 11, 35, 47, 51, 57, 69, 75, 111, 115, 119, 123, 128}));
  h.Draw(Tile(101));
  EXPECT_FALSE(h.CanRiichi());
}

TEST(internal_hand, Opens) {
  auto h = Hand(HandParams("m1,m1,m2,m3,m4,m5,m6,m7,m8,m9,m9,m9,m9"));
  auto chis =
      h.PossibleOpensAfterOthersDiscard(Tile("m1", 2), RelativePos::kLeft);
  h.ApplyOpen(std::move(chis.at(0)));
  const auto opens = h.Opens();
  EXPECT_EQ(opens.size(), 1);
  EXPECT_EQ(opens.front().Type(), OpenType::kChi);
}

TEST(internal_hand, Riichi) {
  auto h = Hand(HandParams("m1,m1,m1,m2,m3,m4,m5,m6,m7,m8,m9,m9,m9"));
  h.Draw(Tile("rd"));
  EXPECT_FALSE(h.IsUnderRiichi());
  h.Riichi();
  EXPECT_TRUE(h.IsUnderRiichi());
}

TEST(internal_hand, PossibleDiscardsAfterRiichi) {
  auto h = Hand(HandParams("m1,m1,m1,m2,m3,m4,m5,m6,m7,m8,m9,m9,m9"));
  h.Draw(Tile("rd"));
  h.Riichi();
  auto possible_discards = h.PossibleDiscardsJustAfterRiichi();
  EXPECT_EQ(possible_discards.size(), 4);
  auto HasType = [&](TileType tt) {
    return std::find_if(possible_discards.begin(), possible_discards.end(),
                        [&](const auto& x) { return x.first.Type() == tt; }) !=
           possible_discards.end();
  };
  EXPECT_TRUE(HasType(TileType::kRD));
  EXPECT_TRUE(HasType(TileType::kM2));
  EXPECT_TRUE(HasType(TileType::kM5));
  EXPECT_TRUE(HasType(TileType::kM8));
}

TEST(internal_hand, ToString) {
  auto h = Hand(HandParams("m1,m1,m1,m2,m3,m4,m5,m6,m7,m8,m9,m9,m9"));
  EXPECT_EQ(h.ToString(), "m1,m1,m1,m2,m3,m4,m5,m6,m7,m8,m9,m9,m9");
  EXPECT_EQ(h.ToString(true),
            "m1(0),m1(1),m1(2),m2(0),m3(0),m4(0),m5(0),m6(0),m7(0),m8(0),m9(0),"
            "m9(1),m9(2)");
  auto possible_opens =
      h.PossibleOpensAfterOthersDiscard(Tile("m5", 3), RelativePos::kLeft);
  h.ApplyOpen(std::move(possible_opens.front()));
  EXPECT_EQ(h.ToString(), "m1,m1,m1,m2,m3,m4,m5,m8,m9,m9,m9,[m5,m6,m7]");
  EXPECT_EQ(h.ToString(true),
            "m1(0),m1(1),m1(2),m2(0),m3(0),m4(0),m5(0),m8(0),m9(0),m9(1),m9(2),"
            "[m5(3),m6(0),m7(0)]");
  h.Discard(Tile("m1", 0));
  h.Draw(Tile("m9", 3));
  possible_opens = h.PossibleOpensAfterDraw();
  h.ApplyOpen(std::move(possible_opens.front()));
  EXPECT_EQ(h.ToString(), "m1,m1,m2,m3,m4,m5,m8,[m5,m6,m7],[m9,m9,m9,m9]c");
  EXPECT_EQ(h.ToString(true),
            "m1(1),m1(2),m2(0),m3(0),m4(0),m5(0),m8(0),[m5(3),m6(0),m7(0)],[m9("
            "0),m9(1),m9(2),m9(3)]c");
}

TEST(internal_hand, LastTileAdded) {
  auto h = Hand(HandParams("m1,m1,m1,m2,m3,m4,m5,m6,m7,m8,m9,m9,m9"));
  EXPECT_TRUE(h.LastTileAdded() == std::nullopt);
  h.Draw(Tile("m1", 3));
  EXPECT_EQ(h.LastTileAdded(), Tile("m1", 3));
  h.Discard(Tile("m1", 0));
  EXPECT_TRUE(h.LastTileAdded() == std::nullopt);
  auto opens =
      h.PossibleOpensAfterOthersDiscard(Tile("m2", 3), RelativePos::kLeft);
  h.ApplyOpen(std::move(opens.front()));
  EXPECT_TRUE(h.LastTileAdded() == Tile("m2", 3));
  h.Discard(Tile("m1", 1));
  EXPECT_TRUE(h.LastTileAdded() == std::nullopt);
}

TEST(internal_hand, Ron) {
  auto h = Hand(HandParams("m1,m1,m1,m2,m3,m4,m5,m6,m7,m8,m9,m9,m9"));
  h.Ron(Tile("m1", 3));
  EXPECT_EQ(h.stage(), HandStage::kAfterRon);
  EXPECT_EQ(h.LastTileAdded(), Tile("m1", 3));
}

TEST(internal_hand, Tsumo) {
  auto h = Hand(HandParams("m1,m1,m1,m2,m3,m4,m5,m6,m7,m8,m9,m9,m9"));
  h.Draw(Tile("m1", 3));
  h.Tsumo();
  EXPECT_EQ(h.stage(), HandStage::kAfterTsumo);
  EXPECT_EQ(h.LastTileAdded(), Tile("m1", 3));

  // after kan
  h = Hand(HandParams("m1,m1,m1,m2,m3,m4,m5,m6,m7,m8,m9,m9,m9"));
  h.Draw(Tile("m9", 3));
  auto possible_opens = h.PossibleOpensAfterDraw();
  h.ApplyOpen(std::move(possible_opens.front()));
  h.Draw(Tile("m1", 3));
  h.Tsumo();
  EXPECT_EQ(h.stage(), HandStage::kAfterTsumoAfterKan);
  EXPECT_EQ(h.LastTileAdded(), Tile("m1", 3));
}

// TEST(internal_hand, EvalScore) {
//     auto h =
//     Hand(HandParams("m1,m1,m1,m2,m3,m4,s3,s3,p2,p2,sw,sw,sw").Tsumo("p2"));
//
//     auto score =
//     h.EvalScore(WinStateInfo().PrevalentWind(Wind::kSouth).IsBottom(true)
//             .IsFirstTsumo(false).Dora({{TileType::kM1, 1}}));
//
//     EXPECT_EQ(score.HasYaku(Yaku::kPrevalentWindSouth),
//     std::make_optional(1)); EXPECT_EQ(score.HasYaku(Yaku::kBottomOfTheSea),
//     std::make_optional(1)); EXPECT_EQ(score.HasYaku(Yaku::kDora),
//     std::make_optional(3));
// }

TEST(internal_hand, IsTenpai) {
  // テンパイ
  auto h = Hand(HandParams("m1,m1,m1,m2,m3,m4,m5,m6,m7,m8,m9,m9,m9"));
  EXPECT_TRUE(h.IsTenpai());
  // テンパイでない
  h = Hand(HandParams("m1,m1,m1,m2,m3,m4,m5,m6,m7,m8,m9,m9,p9"));
  EXPECT_FALSE(h.IsTenpai());
  // 国士無双
  h = Hand(HandParams("m1,m9,p1,p1,p9,s1,s9,ew,sw,ww,nw,wd,rd"));
  EXPECT_TRUE(h.IsTenpai());
  // 国士十三面待ち
  h = Hand(HandParams("m1,m9,p1,p9,s1,s9,ew,sw,ww,nw,wd,gd,rd"));
  EXPECT_TRUE(h.IsTenpai());
  // 鳴きの部分を含めて5枚目を使っている（天鳳では聴牌としてみとめられる）
  // https://tenhou.net/man/
  h = Hand(HandParams("m1")
               .Pon("m1,m1,m1")
               .Pon("p1,p1,p1")
               .Pon("s1,s1,s1")
               .Pon("m9,m9,m9"));
  EXPECT_TRUE(h.IsTenpai());
  h = Hand(
      HandParams("m1,m3,p1,p2,p3,p4,p5,p6,p7,p7").KanOpened("m2,m2,m2,m2"));
  EXPECT_TRUE(h.IsTenpai());
  h = Hand(HandParams("m1,m3,p1,p2,p3,p4,p5,p6,p7,p7")
               .KanClosed("m2,m2,m2,m2"));  // TODO(sotetsuk): 大明槓は副露?
  EXPECT_TRUE(h.IsTenpai());
  // 純手牌（手牌-副露牌）で4枚使って5枚目を待っている（これは聴牌でない）
  h = Hand(HandParams("m1,m1,m1,m1,p1,p2,p3,p4,p5,p6,p7,p8,p9"));
  EXPECT_FALSE(h.IsTenpai());
  h = Hand(HandParams("m1,m1,m1,m1,m2,m3,m4,m4,m4,m4,rd,rd,rd"));
  EXPECT_FALSE(h.IsTenpai());
}

TEST(internal_hand, CanTakeTenpai) {
  auto h =
      Hand(HandParams("m5,m6,s2,s3,s4,p9,p9").Pon("m8,m8,m8").Chi("s5,s6,s7"));
  auto t = Tile::Create({"s3", "s4", "s5"});
  auto c = Chi::Create(t, Tile("s5", 0));
  h.ApplyOpen(c);
  EXPECT_EQ(h.CanTakeTenpai(),
            false);  // s2は喰い替えで切れないのでテンパイは取れない
}
