#ifndef MAHJONG_GAME_SUMMARIZER_H
#define MAHJONG_GAME_SUMMARIZER_H

#include "state.h"

namespace mj
{
    class Game_Summarizer
    {
    public:
        [[nodiscard]] const int& n_game() const;
        void Add(GameResult game_result);
        void show() const;
    private:
        int n_game_ = 0;
    };
}
#endif //MAHJONG_GAME_SUMMARIZER_H
