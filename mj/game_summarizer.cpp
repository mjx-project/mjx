#include "game_summarizer.h"

namespace mj
{
    const int& Game_Summarizer::n_game() const {
        return n_game_;
    }

    void Game_Summarizer::Add(GameResult game_result) {
        n_game_++;
    }

    void Game_Summarizer::show() const {
        std::cout << n_game_ << std::endl;
    }
}  // namespace mj
