#include "game_result_summarizer.h"

namespace mj
{
    const int& GameResultSummarizer::n_game() const {
        return n_game_;
    }

    void GameResultSummarizer::Add(GameResult game_result) {
        n_game_++;
    }

    void GameResultSummarizer::show() const {
        std::cout << n_game_ << std::endl;
    }
}  // namespace mj
