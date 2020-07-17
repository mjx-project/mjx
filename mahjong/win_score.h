#ifndef MAHJONG_WIN_SCORE_H
#define MAHJONG_WIN_SCORE_H

#include "types.h"
#include <array>
#include <map>
#include <utility>

namespace mj {

    class ScoreTable {
        using Fan = int;
        using Fu = int;
    public:
        std::map<std::pair<Fan,Fu>, int> dealer_tsumo, dealer_ron, non_dealer_ron;
        std::map<std::pair<Fan,Fu>, std::pair<int,int>> non_dealer_tsumo;
        ScoreTable();
    };

    class WinningScore {
    private:
        std::map<Yaku,int> yaku_;
        std::map<Yaku,bool> yakuman_;
        std::optional<int> fu_;
    public:
        [[nodiscard]] const std::map<Yaku,int>& yaku() const noexcept ;
        [[nodiscard]] const std::map<Yaku,bool>& yakuman() const noexcept ;
        [[nodiscard]] std::optional<int> fu() const noexcept ;
        [[nodiscard]] int total_fan() const noexcept ;
        [[nodiscard]] std::optional<int> HasYaku(Yaku yaku) const noexcept ;
        [[nodiscard]] bool HasYakuman(Yaku yakuman) const noexcept ;
        void AddYaku(Yaku yaku, int fan) noexcept ;
        void AddYakuman(Yaku yaku) noexcept ;
        void set_fu(int fu) noexcept ;
        [[nodiscard]] bool RequireFan() const noexcept ;
        [[nodiscard]] bool RequireFu() const noexcept ;
        [[nodiscard]] std::array<int,4> Payment(int winner, int dealer, std::optional<int> catched) const noexcept ;
    };

} // namespace mj

#endif //MAHJONG_WIN_SCORE_H
