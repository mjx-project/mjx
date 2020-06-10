#ifndef MAHJONG_BLOCK_H
#define MAHJONG_BLOCK_H

#include <cstdint>
#include <array>
#include <vector>

namespace mj {
    class Block {
    public:
        Block() = default;
        Block(const std::vector<std::uint8_t> &block);
        [[nodiscard]] std::uint32_t Size() const noexcept;
        [[nodiscard]] std::uint32_t Hash() const noexcept;
        [[nodiscard]] std::vector<std::uint8_t> ToVector() const noexcept;
        [[nodiscard]] std::string ToString() const noexcept;
        // comp
        bool operator== (const Block &right) const noexcept;
        bool operator!= (const Block &right) const noexcept;
        bool operator< (const Block &right) const noexcept;
        bool operator<= (const Block &right) const noexcept;
        bool operator> (const Block &right) const noexcept;
        bool operator>= (const Block &right) const noexcept;
        // factory method
        static std::vector<Block> Build(const std::array<std::uint8_t, 34> &arr);
        static std::vector<Block> Build(const std::string &blocks_str);  // e.g., "2,3,111,222"
        // utils
        static std::string BlocksToString(const std::vector<mj::Block> &blocks) noexcept;
    private:
        std::uint32_t hash_ = 0;
    };
}  // namespace mj

#endif //MAHJONG_BLOCK_H
