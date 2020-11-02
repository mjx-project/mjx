#ifndef MAHJONG_GAME_RESULT_SUMMARIZER_H
#define MAHJONG_GAME_RESULT_SUMMARIZER_H

#include "state.h"

namespace mj
{
    class GameResultSummarizer
    {
    public:
        GameResultSummarizer(const GameResultSummarizer&) = delete;
        GameResultSummarizer& operator=(GameResultSummarizer&) = delete;
        GameResultSummarizer(GameResultSummarizer&&) = delete;
        GameResultSummarizer operator=(GameResultSummarizer&&) = delete;

        static const GameResultSummarizer& instance();

        struct PlayerPerformance
        {
            int num_games = 0;
            std::map<int, int> num_ranking = {{1, 0}, {2, 0}, {3, 0}, {4, 0}};
            std::optional<double> avg_ranking = std::nullopt;
            // For the definition of stable dan (安定段位, see these materials:
            //   - Appendix C. https://arxiv.org/abs/2003.13590
            //   - https://tenhou.net/man/#RANKING
            std::optional<double> stable_dan = std::nullopt;
        };
        [[nodiscard]] int num_games() const;
        [[nodiscard]] const PlayerPerformance& player_performance(const PlayerId& player_id) const;
        [[nodiscard]] std::string string() const;
        void Add(GameResult&& game_result);
    private:
        GameResultSummarizer() = default;
        ~GameResultSummarizer() = default;
        int num_games_ = 0;
        std::map<PlayerId, PlayerPerformance> player_performances_;
        static std::optional<double> avg_ranking(const std::map<int, int>& num_ranking);
        static std::optional<double> stable_dan(const std::map<int, int>& num_ranking);
    };
}
#endif //MAHJONG_GAME_RESULT_SUMMARIZER_H
