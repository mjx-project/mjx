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
