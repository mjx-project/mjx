#ifndef MAHJONG_WALL_H
#define MAHJONG_WALL_H

#include "vector"
#include "tile.h"
#include "hand.h"

namespace mj
{
    class Wall
    {
        /*
         * 136 tiles, indexed [0, 135]
         *  - [0, 51] (13*4=52): initial hands of 4 players 配牌
         *  - [52, 121] (70): draws ツモ
         *  - [122, 125] (4): kan draws 嶺上牌
         *  - [126] (1):  dora ドラ
         *  - [127, 130] (4): kan doras カンドラ
         *  - [131] (1): ura dora 裏ドラ
         *  - [132, 135] (4): kan ura doras カンドラ裏
         */
    public:
        explicit Wall(std::uint32_t seed = 9999);
        [[nodiscard]] std::array<std::unique_ptr<Hand>, 4> initial_hands() const;
        Tile Draw();
    private:
        std::uint32_t seed;
        std::unique_ptr<std::vector<Tile>> tiles;
        std::vector<Tile>::const_iterator itr_curr_draw;
        std::vector<Tile>::const_iterator itr_draw_end;
        std::vector<Tile>::const_iterator itr_curr_kan_draw;
        std::vector<Tile>::const_iterator itr_kan_draw_end;
        std::vector<Tile>::const_iterator itr_dora_begin;
        std::vector<Tile>::const_iterator itr_ura_dora_begin;
    };
}  // namespace mj

#endif //MAHJONG_WALL_H
