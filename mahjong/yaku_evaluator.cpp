#include "yaku_evaluator.h"

namespace mj
{

    YakuEvaluator::YakuEvaluator(const WinningHandCache &win_cache) : win_cache_(win_cache){ }

    std::vector<Yaku> YakuEvaluator::Apply(Hand &hand) {
        std::vector<Yaku> yakus;
        std::uint64_t yaku_bit = YakuBit(hand);
        return yakus;
    }

    std::uint64_t mj::YakuEvaluator::YakuBit(Hand &hand) {
        return 0;
    }
}