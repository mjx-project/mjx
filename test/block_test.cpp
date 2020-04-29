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
    auto v = Block::build(a);
    EXPECT_EQ(v.size(), 3);
    EXPECT_EQ(v.at(0).to_string(), "3");
    EXPECT_EQ(v.at(1).to_string(), "111");
    EXPECT_EQ(v.at(2).to_string(), "2222");
    a = {0,0,0,0,0,0,0,0,2,
        1,1,1,1,1,1,1,1,1,
        1,1,1,0,0,0,0,0,0,
        0,0,0,0,0,0, 0};
    v = Block::build(a);
    EXPECT_EQ(v.size(), 3);
    EXPECT_EQ(v.at(0).to_string(), "2");
    EXPECT_EQ(v.at(1).to_string(), "111");
    EXPECT_EQ(v.at(2).to_string(), "111111111");
    a = {1,0,0,0,0,0,0,0,1,
         1,0,0,0,0,0,0,0,1,
         1,0,0,0,0,0,0,0,1,
         1,1,1,1,1,1,1};
    v = Block::build(a);
    EXPECT_EQ(v.size(), 13);

    // From string
    auto blocks = Block::build("2,3,111,222");
    EXPECT_EQ(Block::blocks_to_string(blocks), "2,3,111,222");
}

TEST(block, size)
{
    EXPECT_EQ(Block({1}).size(), 1);
    EXPECT_EQ(Block({3}).size(), 3);
    EXPECT_EQ(Block({1,1,1}).size(), 3);
    EXPECT_EQ(Block({3,1,1}).size(), 5);
    EXPECT_EQ(Block({1,1,3}).size(), 5);
    EXPECT_EQ(Block({3,1,1,1,1,1,1,1,3}).size(), 13);
}

TEST(block, hash)
{
    EXPECT_EQ(Block({1}).hash(), 1);
    EXPECT_EQ(Block({3}).hash(), 3);
    EXPECT_EQ(Block({1,1,1}).hash(), 31);
    EXPECT_EQ(Block({3,1,1}).hash(), 33);
    EXPECT_EQ(Block({1,1,3}).hash(), 81);
}

TEST(block, block)
{
    EXPECT_EQ(Block({1}).block(), std::vector<std::uint8_t>({1}));
    EXPECT_EQ(Block({3}).block(), std::vector<std::uint8_t>({3}));
    EXPECT_EQ(Block({1,1,1}).block(), std::vector<std::uint8_t>({1,1,1}));
    EXPECT_EQ(Block({3,1,1}).block(), std::vector<std::uint8_t>({3,1,1}));
    EXPECT_EQ(Block({1,1,3}).block(), std::vector<std::uint8_t>({1,1,3}));
    EXPECT_EQ(Block({3,1,1,1,1,1,1,1,3}).block(), std::vector<std::uint8_t>({3,1,1,1,1,1,1,1,3}));
}

TEST(block, to_string)
{
    EXPECT_EQ(Block({1}).to_string(), "1");
    EXPECT_EQ(Block({3}).to_string(), "3");
    EXPECT_EQ(Block({1,1,1}).to_string(), "111");
    EXPECT_EQ(Block({3,1,1}).to_string(), "311");
    EXPECT_EQ(Block({1,1,3}).to_string(), "113");
    EXPECT_EQ(Block({3,1,1,1,1,1,1,1,3}).to_string(), "311111113");
}
