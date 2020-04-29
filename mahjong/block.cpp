#include <sstream>
#include <iostream>
#include "block.h"

namespace mj
{
    mj::Block::Block(const std::vector <std::uint8_t> &block) {
        std::uint32_t base_ = 1;
        for (const auto e : block) {
            hash_ += e * base_;
            base_ *= 5;
        }
    }

    std::uint32_t mj::Block::size() const noexcept {
        std::uint32_t s = 0;
        auto h = hash_;
        while (h != 0) {
            s += h % 5;
            h /= 5;
        }
        return s;
    }

    std::uint32_t mj::Block::hash() const noexcept {
        return hash_;
    }

    std::vector<std::uint8_t> mj::Block::block() const noexcept {
        auto v = std::vector<std::uint8_t>();
        auto h = hash_;
        while (h != 0) {
            v.push_back(h % 5);
            h /= 5;
        }
        return v;
    }

    std::string mj::Block::to_string() const noexcept {
        auto v = block();
        std::ostringstream os;
        for (const auto e : v) os << static_cast<int>(e);
        return os.str();
    }

    std::vector<Block> Block::build(const std::array<std::uint8_t, 34> &arr) {
        auto v = std::vector<Block>();
        std::vector<std::uint8_t> b = {};
        auto push = [&]() { v.emplace_back(b); b = {}; };
        auto build_numbers = [&] (int s, int e) {
            for (int i = s; i < e; ++i) {
                if (arr[i] == 0) {
                    if(!b.empty()) push();
                } else {
                    b.push_back(arr[i]);
                }
            }
            if (!b.empty()) push();
        };
        build_numbers(0, 9);    // manzu
        build_numbers(9, 18);   // pinzu
        build_numbers(18, 27);  // souzu
        for (int i = 27; i < 34; ++i) if (arr[i] != 0) v.push_back(Block({arr[i]}));  // for winds and dragons
        std::sort(v.begin(), v.end());
        return v;
    }

    bool Block::operator==(const Block &right) const noexcept {
        return hash() == right.hash();
    }

    bool Block::operator!=(const Block &right) const noexcept {
        return hash() != right.hash();
    }

    bool Block::operator<(const Block &right) const noexcept {
        return hash() < right.hash();
    }

    bool Block::operator<=(const Block &right) const noexcept {
        return hash() <= right.hash();
    }

    bool Block::operator>(const Block &right) const noexcept {
        return hash() > right.hash();
    }

    bool Block::operator>=(const Block &right) const noexcept {
        return hash() >= right.hash();
    }

    std::vector<Block> Block::build(const std::string &blocks_str) {
        auto blocks = std::vector<Block>();
        std::vector<std::uint8_t> v;
        for (const auto c: blocks_str) {
            if (c == ',') {
                blocks.emplace_back(v);
                v.clear();
                continue;
            }
            v.push_back(static_cast<std::uint8_t>(c - '0'));
        }
        blocks.emplace_back(v);
        return blocks;
    }

    std::string Block::blocks_to_string(const std::vector<mj::Block> &blocks) noexcept {
        std::ostringstream os;
        for (const auto &b : blocks) {
            os << b.to_string() << ",";
        }
        std::string s = os.str();
        return s.substr(0, s.size() - 1);
    }
}  // namesapce mj