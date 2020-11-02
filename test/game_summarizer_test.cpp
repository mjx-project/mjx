#include "gtest/gtest.h"
#include <mj/game_result_summarizer.h>
using namespace mj;

TEST(GameResultSummarizer, Add)
{
    GameResultSummarizer& summarizer = GameResultSummarizer::instance();
    summarizer.Initialize();
    EXPECT_EQ(summarizer.num_games(), 0);
    summarizer.Add(GameResult{0, {{"A", 1}, {"B", 2}, {"C", 3}, {"D", 4}}});
    EXPECT_EQ(summarizer.num_games(), 1);
}

TEST(GameResultSummarizer, player_performance)
{
    // avg ranking, stable dan
    GameResultSummarizer& summarizer = GameResultSummarizer::instance();
    summarizer.Initialize();
    summarizer.Add(GameResult{0, {{"A", 1}, {"B", 2}, {"C", 3}, {"D", 4}}});
    summarizer.Add(GameResult{0, {{"A", 2}, {"B", 3}, {"C", 4}, {"D", 1}}});
    summarizer.Add(GameResult{0, {{"A", 3}, {"B", 4}, {"C", 1}, {"D", 2}}});
    summarizer.Add(GameResult{0, {{"A", 4}, {"B", 1}, {"C", 2}, {"D", 3}}});
    for (const auto& player_id: {"A", "B", "C", "D"}) {
        EXPECT_EQ(summarizer.player_performance(player_id).avg_ranking, 2.5);
        EXPECT_EQ(summarizer.player_performance(player_id).stable_dan, 5.0);
    }
    std::cout << summarizer.string() << std::endl;
}
