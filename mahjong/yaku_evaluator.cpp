#include "yaku_evaluator.h"

namespace mj
{
    YakuEvaluator::YakuEvaluator() : win_cache_() {}

    bool YakuEvaluator::Has(const Hand& hand) const noexcept {
        return true;
    }

    std::vector<Yaku> YakuEvaluator::Eval(const Hand& hand) const noexcept {
        return std::vector<Yaku>();
    }
}