#ifndef MAHJONG_WIN_CACHE_H
#define MAHJONG_WIN_CACHE_H

#include <memory>
#include <unordered_map>

namespace mj
{
    class WinningHandCache
    {
    public:
        WinningHandCache();
        [[nodiscard]] std::size_t Size() const noexcept ;
        [[nodiscard]] bool Has(const std::string &s) const noexcept ;
        [[nodiscard]] std::uint64_t Yaku(const std::string &s) const;
        // utils
        static void PrepareWinCache();
        void ShowStats(std::uint64_t yaku_bit, const std::string &yaku_name);
    private:
        std::unique_ptr<std::unordered_map<std::string, std::uint64_t>> cache_;
        void Load();
    };
}

#endif //MAHJONG_WIN_CACHE_H
