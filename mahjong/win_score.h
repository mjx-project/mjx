#ifndef MAHJONG_WIN_SCORE_H
#define MAHJONG_WIN_SCORE_H

#include "types.h"
#include <map>
#include <utility>

namespace mj {

    class WinningScore {
    private:
        std::map<Yaku,int> yaku_;
        std::map<Yaku,bool> yakuman_;
        std::optional<int> fu_;
    public:
        void AddYaku(Yaku yaku, int fan) noexcept ;
        void AddYakuman(Yaku yaku) noexcept ;
        bool RequireFan() const noexcept ;
        bool RequireFu() const noexcept ;
        void SetFu(int fu) noexcept ;
        std::optional<int> HasYaku(Yaku yaku) const noexcept ;
        bool HasYakuman(Yaku yakuman) const noexcept ;
    };

} // namespace mj

#endif //MAHJONG_WIN_SCORE_H
