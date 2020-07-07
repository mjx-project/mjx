#include <cassert>
#include <array>
#include "wall.h"

namespace mj
{
    Wall::Wall(std::uint32_t seed)
            : seed(seed),
              tiles(std::make_unique<std::vector<Tile>>(Tile::CreateAllShuffled(seed))),
              itr_curr_draw(tiles->cbegin() + 52), itr_draw_end(tiles->cbegin() + 122),
              itr_curr_kan_draw(tiles->cbegin() + 122), itr_kan_draw_end(tiles->cbegin() + 130),
              itr_dora_begin(tiles->cbegin() + 126), itr_ura_dora_begin(tiles->cbegin() + 131)
    {}

    Tile Wall::Draw() {
        assert(itr_curr_draw != itr_draw_end);
        auto drawn_tile = *itr_curr_draw;
        itr_curr_draw++;
        return drawn_tile;
    }

    std::array<std::unique_ptr<Hand>, 4> Wall::initial_hands() const {
        std::array<std::unique_ptr<Hand>, 4> hands = {
                std::make_unique<Hand>(tiles->cbegin(), tiles->cbegin() + 13),
                std::make_unique<Hand>(tiles->cbegin() + 13, tiles->cbegin() + 26),
                std::make_unique<Hand>(tiles->cbegin() + 26, tiles->cbegin() + 39),
                std::make_unique<Hand>(tiles->cbegin() + 39, tiles->cbegin() + 52)
        };
        return hands;
    }
}  // namespace mj
