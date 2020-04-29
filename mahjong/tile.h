#ifndef MAHJONG_TILE_H
#define MAHJONG_TILE_H

#include "consts.h"

namespace mj
{
    class Tile
    {
    public:
        Tile() = delete;
        explicit Tile(std::uint8_t tile_id);
        explicit Tile(tile_type tile_type, std::uint8_t offset = 0);
        explicit Tile(const std::string &tile_type_str, std::uint8_t offset = 0);

        static std::vector<Tile> create(const std::vector<tile_id> &vector) noexcept;
        static std::vector<Tile> create(const std::vector<tile_type> &vector) noexcept;
        static std::vector<Tile> create(const std::vector<std::string> &vector) noexcept;
        static std::vector<Tile> create_all() noexcept;

        [[nodiscard]] tile_id id() const noexcept;  // 0 ~ 135
        [[nodiscard]] tile_type type() const noexcept;  // 0 ~ 33
        [[nodiscard]] std::uint8_t type_uint() const noexcept;  // 0 ~ 33
        [[nodiscard]] tile_set_type color() const noexcept;
        [[nodiscard]] std::uint8_t num() const noexcept;  // m1 => 1

        [[nodiscard]] bool is(std::uint8_t n) const noexcept;
        [[nodiscard]] bool is(tile_type tile_type) const noexcept;
        [[nodiscard]] bool is(tile_set_type tile_set_type) const noexcept;
        [[nodiscard]] bool is_red5() const;

        bool operator== (const Tile & right) const noexcept;
        bool operator!= (const Tile & right) const noexcept;
        bool operator< (const Tile & right) const noexcept;
        bool operator<= (const Tile & right) const noexcept;
        bool operator> (const Tile & right) const noexcept;
        bool operator>= (const Tile & right) const noexcept;

        [[nodiscard]] std::string to_string() const noexcept;  // tile_type::ew => "ew"
        [[nodiscard]] std::string to_char() const noexcept;  // tile_type::ew => 東 (East)
        [[nodiscard]] std::string to_unicode() const noexcept;  // tile_type::ew => 🀀

        [[nodiscard]] bool is_valid() const noexcept;

    private:
        tile_id tile_id_;  // 0 ~ 135
        static tile_type str2type(const std::string &s) noexcept;
    };

    struct HashTile {
        std::size_t operator()(const Tile &t) const noexcept {
            return std::hash<int>{}(t.id());
        }
    };
}  // namespace mj

#endif //MAHJONG_TILE_H
