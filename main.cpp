#include <iostream>
#include <mj/mj.h>
#include "mj/agent_grpc_server.h"
#include "mj/agent_grpc_client.h"
#include "mj/agent_grpc_server_impl_rule_based.h"

using namespace mj;

int main(int argc, char* argv[]) {
    std::cout << argc << std::endl;
    assert(argc == 2);
    if(*argv[1]=='0'){
        AgentGrpcServer server(std::make_unique<AgentGrpcServerImplRuleBased>());
        server.RunServer("0.0.0.0:50051");
    }
    else{
        auto channel = grpc::CreateChannel("localhost:50051",grpc::InsecureChannelCredentials());
        const std::vector<std::shared_ptr<Agent>> agents = {
                std::make_shared<AgentGrpcClient>("agent01", channel),
                std::make_shared<AgentGrpcClient>("agent02", channel),
                std::make_shared<AgentGrpcClient>("agent03", channel),
                std::make_shared<AgentGrpcClient>("agent04", channel),
        };
        for(auto thread_num : {1, 2, 3, 4, 10}){
            auto start = std::chrono::system_clock::now();
            Environment::ParallelRunGame(100, thread_num, agents);
            auto end = std::chrono::system_clock::now();
            std::cout <<  thread_num << " " << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() << std::endl;
        }
    }
    return 0;
}
