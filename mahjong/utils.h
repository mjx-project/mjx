#ifndef MAHJONG_UTILS_H
#define MAHJONG_UTILS_H

#include "type_traits"

namespace mj
{
    // ref. Effective Modern C++
    template<typename T>
    constexpr auto toUType(T enumerator) noexcept  {
        return static_cast<std::underlying_type_t<T>>(enumerator);
    }
}  // namespace mj

#endif //MAHJONG_UTILS_H
