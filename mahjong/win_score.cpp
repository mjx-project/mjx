#include "win_score.h"

#include "types.h"
#include <utility>

namespace mj {

    void WinningScore::AddYaku(Yaku yaku, int fan) noexcept {
        yaku_[yaku] = fan;
    }

    void WinningScore::AddYakuman(Yaku yakuman) noexcept {
        yakuman_[yakuman] = true;
    }

    bool WinningScore::RequireFan() const noexcept {
        return yakuman_.empty();
    }

    bool WinningScore::RequireFu() const noexcept {
        if (!yakuman_.empty()) return false;
        int total_fan = 0;
        for (auto& [yaku, fan] : yaku_) {
            total_fan += fan;
        }
        return total_fan <= 4;
    }

    void WinningScore::SetFu(int fu) noexcept {
        fu_ = fu;
    }

    std::optional<int> WinningScore::HasYaku(Yaku yaku) const noexcept {
        if (yaku_.count(yaku)) return yaku_.at(yaku);
        return std::nullopt;
    }
    bool WinningScore::HasYakuman(Yaku yakuman) const noexcept {
        return yakuman_.count(yakuman);
    }

} // namespace mj
