#ifndef MAHJONG_UTILS_H
#define MAHJONG_UTILS_H

#include <random>
#include <iterator>
#include <algorithm>
#include <thread>
#include <initializer_list>

namespace mj
{
    // original assertion
    #define Assert(fmt, ...) \
        assert(fmt || Msg(__VA_ARGS__))

    // 引数なし
    bool Msg(){ return false; }

    // 引数１つ以上
    template < typename ...Args >
    bool Msg(const Args& ...args ){
        std::cout << "Assertion failed: " << std::endl;
        for(const auto& arg : {args...}){
            std::cout << arg << std::endl;
        }
        return false;
    }

    template<typename T>
    bool Any(T target, const std::initializer_list<T> &v) {
        return std::any_of(v.begin(), v.end(), [&target](T elem) { return target == elem; });
    }

    template<typename T>
    bool Any(T target, const std::vector<T> &v) {
        return std::any_of(v.begin(), v.end(), [&target](T elem) { return target == elem; });
    }

    template<typename T, typename F>
    bool Any(const std::vector<T> &v, F && f) {
        return std::any_of(v.begin(), v.end(), [&f](T elem) { return f(elem); });
    }

    // https://stackoverflow.com/questions/6942273/how-to-get-a-random-element-from-a-c-container
    template<typename Iter, typename RandomGenerator>
    Iter SelectRandomly(Iter start, Iter end, RandomGenerator& g) {
        std::uniform_int_distribution<> dis(0, std::distance(start, end) - 1);
        std::advance(start, dis(g));
        return start;
    }

    template<typename Iter>
    Iter SelectRandomly(Iter start, Iter end) {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        return SelectRandomly(start, end, gen);
    }

    // From Effective Modern C++
    template<typename E>
    constexpr auto ToUType(E enumerator) noexcept {
        return static_cast<std::underlying_type_t<E>>(enumerator);
    }

    // A fork from https://github.com/PacktPublishing/The-Modern-Cpp-Challenge
    template<typename RandomAccessIterator, typename F>
    void ptransform(RandomAccessIterator begin, RandomAccessIterator end, F&& f) {
        auto size = std::distance(begin, end);
        std::vector<std::thread> threads;
        const auto thread_count = std::thread::hardware_concurrency();
        auto first = begin;
        auto last = first;
        size /= thread_count;
        for (int i = 0; i < thread_count; ++i) {
            first = last;
            last = (i == thread_count - 1) ? end : first + size;
            threads.emplace_back([first, last, &f](){ std::transform(first, last, first, std::forward<F>(f)); });
        }
        for (auto &t: threads) t.join();
    }
}

#endif //MAHJONG_UTILS_H
