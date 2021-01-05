#include <iostream>
#include <mj/mj.h>
#include "mj/agent_grpc_client.h"
#include "mj/agent_grpc_server.h"
#include "mj/strategy_rule_based.h"

using namespace mj;

int main(int argc, char* argv[]) {
    std::cout << "cnt_args: " <<  argc << std::endl;
    assert(argc == 1 || argc == 3);
    if(argc == 1){
        AgentGrpcServer::RunServer(
                std::make_unique<StrategyRuleBased>(), "0.0.0.0:50051"
        );
    }
    else{
        auto channel = grpc::CreateChannel("localhost:50051",grpc::InsecureChannelCredentials());
        const std::vector<std::shared_ptr<Agent>> agents = {
                std::make_shared<AgentGrpcClient>("agent01", channel),
                std::make_shared<AgentGrpcClient>("agent02", channel),
                std::make_shared<AgentGrpcClient>("agent03", channel),
                std::make_shared<AgentGrpcClient>("agent04", channel),
        };
        auto start = std::chrono::system_clock::now();
        Environment::ParallelRunGame(std::atoi(argv[1]), std::atoi(argv[2]), agents);
        auto end = std::chrono::system_clock::now();
        std::cout << "# games: " << std::atoi(argv[1]) << std::endl;
        std::cout << "# threads: " << std::atoi(argv[2]) << std::endl;
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
        std::cout << "total [sec]: " << ms / 1000.0 << std::endl;
        std::cout << "sec/game: " << ms / 1000.0 / std::atoi(argv[1]) << std::endl;
    }
    return 0;
}
