#ifndef MAHJONG_UTILS_H
#define MAHJONG_UTILS_H

#include <algorithm>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <cassert>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <random>
#include <thread>

namespace mjx::internal {

// original assertion
#define Assert(fmt, ...) assert(fmt || Msg(__VA_ARGS__))

template <typename... Args>
bool Msg(const Args&... args) {
  std::cout << "Assertion failed: " << std::endl;
  for (const auto& str : std::initializer_list<std::string>{args...}) {
    std::cout << str << std::endl;
  }
  return false;
}

template <typename T>
bool Any(T target, const std::initializer_list<T>& v) {
  return std::any_of(v.begin(), v.end(),
                     [&target](T elem) { return target == elem; });
}

template <typename T>
bool Any(T target, const std::vector<T>& v) {
  return std::any_of(v.begin(), v.end(),
                     [&target](T elem) { return target == elem; });
}

template <typename T, typename F>
bool Any(const std::vector<T>& v, F&& f) {
  return std::any_of(v.begin(), v.end(), [&f](T elem) { return f(elem); });
}

// https://stackoverflow.com/questions/6942273/how-to-get-a-random-element-from-a-c-container
template <typename Iter, typename RandomGenerator>
Iter SelectRandomly(Iter start, Iter end, RandomGenerator& g) {
  boost::random::uniform_int_distribution<> dis(
      0, std::distance(start, end) - 1);  // use boost ver instead of std to
                                          // avoid implementation dependency
  std::advance(start, dis(g));
  return start;
}

template <typename RandomGenerator>
double RandomProb(RandomGenerator& g) {
  boost::random::uniform_real_distribution<double> uniform(
      0.0,
      1.0);  // use boost ver instead of std to avoid implementation dependency
  return uniform(g);
}

// From Effective Modern C++
template <typename E>
constexpr auto ToUType(E enumerator) noexcept {
  return static_cast<std::underlying_type_t<E>>(enumerator);
}

// This implementation is from
// https://en.cppreference.com/w/cpp/algorithm/random_shuffle#Possible_implementation
// As the std::shuffle implementation is not specified in the standard, we
// cannot reproduce the results with std::shuffle.
template <class RandomIt, class URBG>
void Shuffle(RandomIt first, RandomIt last, URBG&& g) {
  typedef typename std::iterator_traits<RandomIt>::difference_type diff_t;
  typedef boost::random::uniform_int_distribution<diff_t>
      distr_t;  // use boost ver instead of std to avoid implementation
  // dependency
  typedef typename distr_t::param_type param_t;

  distr_t D;
  diff_t n = last - first;
  for (diff_t i = n - 1; i > 0; --i) {
    using std::swap;
    swap(first[i], first[D(g, param_t(0, i))]);
  }
}

// A fork from https://github.com/PacktPublishing/The-Modern-Cpp-Challenge
template <typename RandomAccessIterator, typename F>
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
    threads.emplace_back([first, last, &f]() {
      std::transform(first, last, first, std::forward<F>(f));
    });
  }
  for (auto& t : threads) t.join();
}
}  // namespace mjx::internal

#endif  // MAHJONG_UTILS_H
