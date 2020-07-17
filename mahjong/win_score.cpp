#include "win_score.h"

#include "types.h"
#include <utility>

namespace mj {

    ScoreTable::ScoreTable() {
        dealer_tsumo = {
                {{1,30},500},
                {{1,40},700},
                {{1,50},800},
                {{1,60},1000},
                {{1,70},1200},
                {{1,80},1300},
                {{1,90},1500},
                {{1,100},1600},

                {{2,20},700},
                {{2,30},1000},
                {{2,40},1300},
                {{2,50},1600},
                {{2,60},2000},
                {{2,70},2300},
                {{2,80},2600},
                {{2,90},2900},
                {{2,100},3200},
                {{2,110},3600},

                {{3,20},1300},
                {{3,25},1600},
                {{3,30},2000},
                {{3,40},2600},
                {{3,50},3200},
                {{3,60},3900},

                {{4,20},2600},
                {{4,25},3200},
                {{4,30},3900}
        };

        dealer_ron = {
                {{1,30},1500},
                {{1,40},2000},
                {{1,50},2400},
                {{1,60},2900},
                {{1,70},3400},
                {{1,80},3900},
                {{1,90},4400},
                {{1,100},4800},
                {{1,110},5300},

                {{2,25},2400},
                {{2,30},2900},
                {{2,40},3900},
                {{2,50},4800},
                {{2,60},5800},
                {{2,70},6800},
                {{2,80},7700},
                {{2,90},8700},
                {{2,100},9600},
                {{2,110},10600},

                {{3,25},4800},
                {{3,30},5800},
                {{3,40},7700},
                {{3,50},9600},
                {{3,60},11600},

                {{4,25},9600},
                {{4,30},11600}
        };

        non_dealer_ron = {
                {{1,30},1000},
                {{1,40},1300},
                {{1,50},1600},
                {{1,60},2000},
                {{1,70},2300},
                {{1,80},2600},
                {{1,90},2900},
                {{1,100},3200},
                {{1,110},3600},

                {{2,25},1600},
                {{2,30},2000},
                {{2,40},2600},
                {{2,50},3200},
                {{2,60},3900},
                {{2,70},4500},
                {{2,80},5200},
                {{2,90},5800},
                {{2,100},6400},
                {{2,110},7100},

                {{3,25},3200},
                {{3,30},3900},
                {{3,40},5300},
                {{3,50},6400},
                {{3,60},7700},
                {{4,25},6400},
                {{4,30},7700}
        };

        non_dealer_tsumo = {
                {{1,30},{300,500}},
                {{1,40},{400,700}},
                {{1,50},{400,800}},
                {{1,60},{500,1000}},
                {{1,70},{600,1200}},
                {{1,80},{700,1300}},
                {{1,90},{800,1500}},
                {{1,100},{800,1600}},

                {{2,20},{400,700}},
                {{2,30},{500,1000}},
                {{2,40},{700,1300}},
                {{2,50},{800,1600}},
                {{2,60},{1000,2000}},
                {{2,70},{1200,2300}},
                {{2,80},{1300,2600}},
                {{2,90},{1500,2900}},
                {{2,100},{1600,3200}},
                {{2,110},{1800,3600}},

                {{3,20},{700,1300}},
                {{3,25},{800,1600}},
                {{3,30},{1000,2000}},
                {{3,40},{1300,2600}},
                {{3,50},{1600,3200}},
                {{3,60},{2000,3900}},

                {{4,20},{1300,2600}},
                {{4,25},{1600,3200}},
                {{4,30},{2000,3900}}
        };
    }

    std::array<int,4> WinningScore::Payment(int player, int dealer, std::optional<int> catched) const noexcept {
        static ScoreTable table;

        int fan = total_fan();
        int fu = this->fu() ? this->fu().value() : 0;

        if (catched) {
            int payment;

            if (player == dealer) {
                // 親に振り込んだとき

                if (!yakuman_.empty()) payment = 48000 * yakuman_.size();
                else if (table.dealer_ron.count({fan, fu})) {
                    payment = table.dealer_ron.at({fan, fu});
                }
                else {
                    if (fan <= 5) payment = 12000;
                    else if (fan <= 7) payment = 18000;
                    else if (fan <= 10) payment = 24000;
                    else if (fan <= 12) payment = 36000;
                    else payment = 48000;
                }
            }
            else {
                // 子に振り込んだとき

                if (!yakuman_.empty()) payment = 32000 * yakuman_.size();
                else if (table.non_dealer_ron.count({fan, fu})) {
                    payment = table.dealer_ron.at({fan, fu});
                }
                else {
                    if (fan <= 5) payment = 8000;
                    else if (fan <= 7) payment = 12000;
                    else if (fan <= 10) payment = 16000;
                    else if (fan <= 12) payment = 24000;
                    else payment = 32000;
                }
            }
            std::array<int,4> ret{0,0,0,0};
            ret[catched.value()] = payment;
            return ret;
        }

        else {
            if (player == dealer) {
                // 親がツモ上がりしたとき
                int payment;
                if (!yakuman_.empty()) payment = 16000 * yakuman_.size();
                else if (table.dealer_tsumo.count({fan, fu})) {
                    payment = table.dealer_tsumo.at({fan, fu});
                }
                else {
                    if (fan <= 5) payment = 4000;
                    else if (fan <= 7) payment = 6000;
                    else if (fan <= 10) payment = 8000;
                    else if (fan <= 12) payment = 12000;
                    else payment = 16000;
                }

                std::array<int,4> ret{payment, payment, payment, payment};
                ret[player] = 0;
                return ret;
            }
            else {
                // 子がツモ上がりしたとき

                int dealer_payment, non_dealer_payment;
                if (!yakuman_.empty()) {
                    dealer_payment = 16000 * yakuman_.size();
                    non_dealer_payment = 8000 * yakuman_.size();
                }
                else if (table.non_dealer_tsumo.count({fan, fu})) {
                    std::tie(non_dealer_payment, dealer_payment) =
                            table.non_dealer_tsumo.at({fan, fu});
                }
                else {
                    if (fan <= 5) non_dealer_payment = 2000, dealer_payment = 4000;
                    else if (fan <= 7) non_dealer_payment = 3000, dealer_payment = 6000;
                    else if (fan <= 10) non_dealer_payment = 4000, dealer_payment = 8000;
                    else if (fan <= 12) non_dealer_payment = 6000, dealer_payment = 12000;
                    else non_dealer_payment = 8000, dealer_payment = 16000;
                }

                std::array<int,4> ret{};
                for (int i = 0; i < 4; ++i) {
                    if (i == player) ret[i] = 0;
                    else if (i == dealer) ret[i] = dealer_payment;
                    else ret[i] = non_dealer_payment;
                }
                return ret;
            }
        }
    }

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
        return total_fan() <= 4;
    }

    void WinningScore::set_fu(int fu) noexcept {
        fu_ = fu;
    }

    std::optional<int> WinningScore::HasYaku(Yaku yaku) const noexcept {
        if (yaku_.count(yaku)) return yaku_.at(yaku);
        return std::nullopt;
    }
    bool WinningScore::HasYakuman(Yaku yakuman) const noexcept {
        return yakuman_.count(yakuman);
    }

    const std::map<Yaku,int>& WinningScore::yaku() const noexcept {
        return yaku_;
    }
    const std::map<Yaku,bool>& WinningScore::yakuman() const noexcept {
        return yakuman_;
    }

    std::optional<int> WinningScore::fu() const noexcept {
        return fu_;
    }

    int WinningScore::total_fan() const noexcept {
        int total_fan = 0;
        for (auto& [yaku, fan] : yaku_) {
            total_fan += fan;
        }
        return total_fan;
    }
} // namespace mj
