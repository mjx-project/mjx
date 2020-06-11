#ifndef MAHJONG_YAKU_EVALUATOR_H
#define MAHJONG_YAKU_EVALUATOR_H

#include "types.h"
#include "win_cache.h"
#include "hand.h"

namespace mj
{
    class YakuEvaluator
    {
    public:
        YakuEvaluator(const WinningHandCache &win_cache);
        std::vector<Yaku> Apply(Hand &hand);
    private:
        const WinningHandCache &win_cache_;
        std::uint64_t YakuBit(Hand &hand);
    };

}  // namespace mj

#endif //MAHJONG_YAKU_EVALUATOR_H
