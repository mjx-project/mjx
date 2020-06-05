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

        static std::vector<Tile> Create(const std::vector<tile_id> &vector) noexcept;
        static std::vector<Tile> Create(const std::vector<tile_type> &vector) noexcept;
        static std::vector<Tile> Create(const std::vector<std::string> &vector) noexcept;
        static std::vector<Tile> CreateAll() noexcept;

        [[nodiscard]] tile_id Id() const noexcept;  // 0 ~ 135
        [[nodiscard]] tile_type Type() const noexcept;  // 0 ~ 33
        [[nodiscard]] std::uint8_t TypeUint() const noexcept;  // 0 ~ 33
        [[nodiscard]] tile_set_type Color() const noexcept;
        [[nodiscard]] std::uint8_t Num() const noexcept;  // m1 => 1

        [[nodiscard]] bool Is(std::uint8_t n) const noexcept;
        [[nodiscard]] bool Is(tile_type tile_type) const noexcept;
        [[nodiscard]] bool Is(tile_set_type tile_set_type) const noexcept;
        [[nodiscard]] bool IsRedFive() const;

        bool operator== (const Tile & right) const noexcept;
        bool operator!= (const Tile & right) const noexcept;
        bool operator< (const Tile & right) const noexcept;
        bool operator<= (const Tile & right) const noexcept;
        bool operator> (const Tile & right) const noexcept;
        bool operator>= (const Tile & right) const noexcept;

        [[nodiscard]] std::string ToString() const noexcept;  // tile_type::ew => "ew"
        [[nodiscard]] std::string ToChar() const noexcept;  // tile_type::ew => 東 (East)
        [[nodiscard]] std::string ToUnicode() const noexcept;  // tile_type::ew => 🀀

        [[nodiscard]] bool IsValid() const noexcept;

    private:
        tile_id tile_id_;  // 0 ~ 135
        static tile_type Str2Type(const std::string &s) noexcept;
    };

    struct HashTile {
        std::size_t operator()(const Tile &t) const noexcept {
            return std::hash<int>{}(t.Id());
        }
    };
}  // namespace mj

#endif //MAHJONG_TILE_H
