#include "gtest/gtest.h"
#include <mj/game_result_summarizer.h>
using namespace mj;

TEST(GameResultSummarizer, Add) {
    GameResultSummarizer game_summalizer;
    EXPECT_EQ(game_summalizer.n_game(), 0);
    game_summalizer.Add(GameResult());
    EXPECT_EQ(game_summalizer.n_game(), 1);
}
