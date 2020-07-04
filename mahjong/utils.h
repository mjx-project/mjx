#ifndef MAHJONG_UTILS_H
#define MAHJONG_UTILS_H

#include "random"
#include "iterator"
#include "algorithm"
#include <initializer_list>

namespace mj
{
    template<typename T>
    bool any_of(T target, std::initializer_list<T> list) {
        return std::any_of(list.begin(), list.end(), [&target](T elem) { return target == elem; });
    }

    template<typename T>
    bool any_of(T target, std::vector<T> list) {
        return std::any_of(list.begin(), list.end(), [&target](T elem) { return target == elem; });
    }

    // https://stackoverflow.com/questions/6942273/how-to-get-a-random-element-from-a-c-container
    template<typename Iter, typename RandomGenerator>
    Iter select_randomly(Iter start, Iter end, RandomGenerator& g) {
        std::uniform_int_distribution<> dis(0, std::distance(start, end) - 1);
        std::advance(start, dis(g));
        return start;
    }

    template<typename Iter>
    Iter select_randomly(Iter start, Iter end) {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        return select_randomly(start, end, gen);
    }

    // From Effective Modern C++
    template<typename E>
    constexpr auto ToUType(E enumerator) noexcept {
        return static_cast<std::underlying_type_t<E>>(enumerator);
    }
}

#endif //MAHJONG_UTILS_H
