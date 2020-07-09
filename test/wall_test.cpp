#include "gtest/gtest.h"
#include "wall.h"

using namespace mj;

TEST(wall, ToString) {
    // 文字列変換後で文字列に被りなし
    auto wall = Wall();
    auto wall_str = wall.ToString(true);
    std::replace(wall_str.begin(), wall_str.end(), ',', ' ');
    std::istringstream iss(wall_str);
    std::unordered_set<std::string> set;
    std::string s;
    while (iss >> s) {
        EXPECT_EQ(set.count(s), 0);
        set.insert(s);
    }
    EXPECT_EQ(set.size(), 136);
}

TEST(wall, Draw) {
    // カンなしで70回ツモが存在する
    auto wall = Wall();
    for (int i = 0; i < 70; ++i) {
        EXPECT_TRUE(wall.HasDrawLeft());
        wall.Draw();
    }
    EXPECT_FALSE(wall.HasDrawLeft());
}

TEST(wall, KanDraw) {
    // カンがあると、その分ツモ数が減る
    auto wall = Wall();
    for (int i = 0; i < 35; ++i) {
        EXPECT_TRUE(wall.HasDrawLeft());
        wall.Draw();
    }
    for (int i = 0; i < 4; ++i) {
        wall.KanDraw();
        wall.AddKanDora();
    }
    for (int i = 0; i < 31; ++i) {
        EXPECT_TRUE(wall.HasDrawLeft());
        wall.Draw();
    }
    EXPECT_FALSE(wall.HasDrawLeft());
}

TEST(wall, doras) {
    auto wall = Wall();
    EXPECT_EQ(wall.doras().size(), 1);
    EXPECT_EQ(wall.ura_doras().size(), 1);
    for (int i = 0; i < 4; ++i) {
        wall.KanDraw();
        wall.AddKanDora();
    }
    EXPECT_EQ(wall.doras().size(), 5);
    EXPECT_EQ(wall.ura_doras().size(), 5);
}