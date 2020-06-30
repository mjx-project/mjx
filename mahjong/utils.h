#ifndef MAHJONG_UTILS_H
#define MAHJONG_UTILS_H

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
}

#endif //MAHJONG_UTILS_H
