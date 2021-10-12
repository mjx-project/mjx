#include <gtest/gtest.h>
#include <mjx/seed_generator.h>

TEST(seed_generator, RandomSeedGenerator) {
  std::vector<mjx::PlayerId> player_ids = {"player_0", "player_1", "player_2",
                                           "player_3"};
  std::unique_ptr<mjx::SeedGenerator> seed_generator =
      std::make_unique<mjx::RandomSeedGenerator>(player_ids);
  std::set<std::uint64_t> seeds;
  std::unordered_map<mjx::PlayerId, int> first_dealer_cnt = {
      {"player_0", 0}, {"player_1", 0}, {"player_2", 0}, {"player_3", 0}};
  int N = 100000;
  for (int i = 0; i < N; ++i) {
    auto [seed, player_ids] = seed_generator->Get();
    seeds.insert(seed);
    first_dealer_cnt[player_ids[0]]++;
  }
  EXPECT_EQ(seeds.size(), N);
  EXPECT_TRUE(first_dealer_cnt.at("player_0") > N / 4 - N / 10);
  EXPECT_TRUE(first_dealer_cnt.at("player_0") < N / 4 + N / 10);
  EXPECT_TRUE(first_dealer_cnt.at("player_1") > N / 4 - N / 10);
  EXPECT_TRUE(first_dealer_cnt.at("player_1") < N / 4 + N / 10);
  EXPECT_TRUE(first_dealer_cnt.at("player_2") > N / 4 - N / 10);
  EXPECT_TRUE(first_dealer_cnt.at("player_2") < N / 4 + N / 10);
  EXPECT_TRUE(first_dealer_cnt.at("player_3") > N / 4 - N / 10);
  EXPECT_TRUE(first_dealer_cnt.at("player_3") < N / 4 + N / 10);
}

TEST(seed_generator, DuplicateRandomSeedGenerator) {
  std::vector<mjx::PlayerId> player_ids = {"player_0", "player_1", "player_2",
                                           "player_3"};
  std::unique_ptr<mjx::SeedGenerator> seed_generator =
      std::make_unique<mjx::DuplicateRandomSeedGenerator>(player_ids);
  std::set<std::uint64_t> seeds;
  std::unordered_map<mjx::PlayerId, int> first_dealer_cnt = {
      {"player_0", 0}, {"player_1", 0}, {"player_2", 0}, {"player_3", 0}};
  int N = 100000;
  for (int i = 0; i < N; ++i) {
    auto [seed, player_ids] = seed_generator->Get();
    seeds.insert(seed);
    first_dealer_cnt[player_ids[0]]++;
  }
  EXPECT_EQ(seeds.size(), N / 4);
  EXPECT_TRUE(first_dealer_cnt.at("player_0") == N / 4);
  EXPECT_TRUE(first_dealer_cnt.at("player_1") == N / 4);
  EXPECT_TRUE(first_dealer_cnt.at("player_2") == N / 4);
  EXPECT_TRUE(first_dealer_cnt.at("player_3") == N / 4);
}
