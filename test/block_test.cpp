#include "gtest/gtest.h"
#include "block.h"

using namespace mj;

TEST(block, Block)
{
    EXPECT_NO_FATAL_FAILURE(Block({1, 2, 2, 1}));
}

TEST(block, build)
{
    std::array<std::uint8_t, 34>
    a = {0,0,0,0,2,2,2,2,0,
         0,0,0,0,0,0,0,0,0,
         0,0,0,0,0,0,1,1,1,
         0,0,0,0,0,3, 0};
    auto v = Block::Build(a);
    EXPECT_EQ(v.size(), 3);
    EXPECT_EQ(v.at(0).ToString(), "3");
    EXPECT_EQ(v.at(1).ToString(), "111");
    EXPECT_EQ(v.at(2).ToString(), "2222");
    a = {0,0,0,0,0,0,0,0,2,
        1,1,1,1,1,1,1,1,1,
        1,1,1,0,0,0,0,0,0,
        0,0,0,0,0,0, 0};
    v = Block::Build(a);
    EXPECT_EQ(v.size(), 3);
    EXPECT_EQ(v.at(0).ToString(), "2");
    EXPECT_EQ(v.at(1).ToString(), "111");
    EXPECT_EQ(v.at(2).ToString(), "111111111");
    a = {1,0,0,0,0,0,0,0,1,
         1,0,0,0,0,0,0,0,1,
         1,0,0,0,0,0,0,0,1,
         1,1,1,1,1,1,1};
    v = Block::Build(a);
    EXPECT_EQ(v.size(), 13);

    // From string
    auto blocks = Block::Build("2,3,111,222");
    EXPECT_EQ(Block::BlocksToString(blocks), "2,3,111,222");
}

TEST(block, size)
{
    EXPECT_EQ(Block({1}).Size(), 1);
    EXPECT_EQ(Block({3}).Size(), 3);
    EXPECT_EQ(Block({1, 1, 1}).Size(), 3);
    EXPECT_EQ(Block({3, 1, 1}).Size(), 5);
    EXPECT_EQ(Block({1, 1, 3}).Size(), 5);
    EXPECT_EQ(Block({3, 1, 1, 1, 1, 1, 1, 1, 3}).Size(), 13);
}

TEST(block, hash)
{
    EXPECT_EQ(Block({1}).Hash(), 1);
    EXPECT_EQ(Block({3}).Hash(), 3);
    EXPECT_EQ(Block({1, 1, 1}).Hash(), 31);
    EXPECT_EQ(Block({3, 1, 1}).Hash(), 33);
    EXPECT_EQ(Block({1, 1, 3}).Hash(), 81);
}

TEST(block, block)
{
    EXPECT_EQ(Block({1}).ToVector(), std::vector<std::uint8_t>({1}));
    EXPECT_EQ(Block({3}).ToVector(), std::vector<std::uint8_t>({3}));
    EXPECT_EQ(Block({1, 1, 1}).ToVector(), std::vector<std::uint8_t>({1, 1, 1}));
    EXPECT_EQ(Block({3, 1, 1}).ToVector(), std::vector<std::uint8_t>({3, 1, 1}));
    EXPECT_EQ(Block({1, 1, 3}).ToVector(), std::vector<std::uint8_t>({1, 1, 3}));
    EXPECT_EQ(Block({3, 1, 1, 1, 1, 1, 1, 1, 3}).ToVector(), std::vector<std::uint8_t>({3, 1, 1, 1, 1, 1, 1, 1, 3}));
}

TEST(block, to_string)
{
    EXPECT_EQ(Block({1}).ToString(), "1");
    EXPECT_EQ(Block({3}).ToString(), "3");
    EXPECT_EQ(Block({1, 1, 1}).ToString(), "111");
    EXPECT_EQ(Block({3, 1, 1}).ToString(), "311");
    EXPECT_EQ(Block({1, 1, 3}).ToString(), "113");
    EXPECT_EQ(Block({3, 1, 1, 1, 1, 1, 1, 1, 3}).ToString(), "311111113");
}
