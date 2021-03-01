#include <iostream>
#include <algorithm>
#include <mjx/mjx.h>
#include "mjx/agent_grpc_client.h"
#include "mjx/agent_grpc_server.h"
#include "mjx/strategy_rule_based.h"

using namespace mjx;

// grpc無:
// ./mahjong.out #game #thread
// grpc有:
// ./mahjong.out {host | #game #thread client}

bool cmdOptionExists(char** begin, char** end, const std::string& option)
{
    return std::find(begin, end, option) != end;
}

int main(int argc, char* argv[]) {
    std::cout << "cnt_args: " <<  argc << std::endl;
    if(cmdOptionExists(argv, argv+argc, "host")){
        AgentGrpcServer::RunServer(
                std::make_unique<StrategyRuleBased>(), "0.0.0.0:50051"
        );
    }
    else{
        std::vector<std::shared_ptr<Agent>> agents;
        if(cmdOptionExists(argv, argv+argc, "client")){
            auto channel_rulebased = grpc::CreateChannel("localhost:50051",grpc::InsecureChannelCredentials());
            agents = {
                    std::make_shared<AgentGrpcClient>("rule-based-0", channel_rulebased),
                    std::make_shared<AgentGrpcClient>("rule-based-1", channel_rulebased),
                    std::make_shared<AgentGrpcClient>("rule-based-2", channel_rulebased),
                    std::make_shared<AgentGrpcClient>("rule-based-3", channel_rulebased),
                    };
        }
        else {
            agents = {
                    std::make_shared<AgentExampleRuleBased>("rule-based-0"),
                    std::make_shared<AgentExampleRuleBased>("rule-based-1"),
                    std::make_shared<AgentExampleRuleBased>("rule-based-2"),
                    std::make_shared<AgentExampleRuleBased>("rule-based-3")
            };
        }
        auto start = std::chrono::system_clock::now();
        auto results = Environment::ParallelRunGame(std::atoi(argv[1]), std::atoi(argv[2]), agents);
        auto end = std::chrono::system_clock::now();
        auto &summarizer = GameResultSummarizer::instance();
        summarizer.Initialize();
        for(auto result: results){
            summarizer.Add(std::move(result));
        }
        std::cout << summarizer.string() << std::endl;
        std::cout << "# games: " << std::atoi(argv[1]) << std::endl;
        std::cout << "# threads: " << std::atoi(argv[2]) << std::endl;
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
        std::cout << "total [sec]: " << ms / 1000.0 << std::endl;
        std::cout << "sec/game: " << ms / 1000.0 / std::atoi(argv[1]) << std::endl;
    }
    return 0;
}
