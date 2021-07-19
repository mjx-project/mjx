#include <mjx/internal/game_result_summarizer.h>

#include "gtest/gtest.h"
using namespace mjx::internal;

TEST(internal_game_result_summarizer, Add) {
  GameResultSummarizer& summarizer = GameResultSummarizer::instance();
  summarizer.Initialize();
  EXPECT_EQ(summarizer.num_games(), 0);
  summarizer.Add(GameResult{0, {{"A", 1}, {"B", 2}, {"C", 3}, {"D", 4}}});
  EXPECT_EQ(summarizer.num_games(), 1);

  // thread-safe test
  summarizer.Initialize();
  std::thread th1([&] {
    for (int i = 0; i < 1000; i++) {
      summarizer.Add(GameResult{0, {{"A", 1}, {"B", 2}, {"C", 3}, {"D", 4}}});
      summarizer.Add(GameResult{0, {{"A", 2}, {"B", 3}, {"C", 4}, {"D", 1}}});
      summarizer.Add(GameResult{0, {{"A", 3}, {"B", 4}, {"C", 1}, {"D", 2}}});
      summarizer.Add(GameResult{0, {{"A", 4}, {"B", 1}, {"C", 2}, {"D", 3}}});
    }
  });
  std::thread th2([&] {
    for (int i = 0; i < 1000; i++) {
      summarizer.Add(GameResult{0, {{"A", 1}, {"B", 2}, {"C", 3}, {"D", 4}}});
      summarizer.Add(GameResult{0, {{"A", 2}, {"B", 3}, {"C", 4}, {"D", 1}}});
      summarizer.Add(GameResult{0, {{"A", 3}, {"B", 4}, {"C", 1}, {"D", 2}}});
      summarizer.Add(GameResult{0, {{"A", 4}, {"B", 1}, {"C", 2}, {"D", 3}}});
    }
  });
  th1.join();
  th2.join();
  EXPECT_EQ(summarizer.num_games(), 8000);
  for (const auto& player_id : {"A", "B", "C", "D"}) {
    EXPECT_EQ(summarizer.player_performance(player_id).avg_ranking, 2.5);
    EXPECT_EQ(summarizer.player_performance(player_id).stable_dan, 5.0);
  }
}

TEST(internal_game_result_summarizer, player_performance) {
  // avg ranking, stable dan
  GameResultSummarizer& summarizer = GameResultSummarizer::instance();
  summarizer.Initialize();
  summarizer.Add(GameResult{0, {{"A", 1}, {"B", 2}, {"C", 3}, {"D", 4}}});
  summarizer.Add(GameResult{0, {{"A", 2}, {"B", 3}, {"C", 4}, {"D", 1}}});
  summarizer.Add(GameResult{0, {{"A", 3}, {"B", 4}, {"C", 1}, {"D", 2}}});
  summarizer.Add(GameResult{0, {{"A", 4}, {"B", 1}, {"C", 2}, {"D", 3}}});
  for (const auto& player_id : {"A", "B", "C", "D"}) {
    EXPECT_EQ(summarizer.player_performance(player_id).avg_ranking, 2.5);
    EXPECT_EQ(summarizer.player_performance(player_id).stable_dan, 5.0);
  }
  std::cout << summarizer.string() << std::endl;
}
