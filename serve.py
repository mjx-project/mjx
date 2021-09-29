import mjx.agent
import _mjx

agent = mjx.agent.RandomAgent()
_mjx.AgentServer.serve(agent, "127.0.0.1:9090", 1, 0, 0)

# agents = {
#     "player_0": mjx.agent.RandomAgent(),
#     "player_1": mjx.agent.RandomAgent(),
#     "player_2": mjx.agent.RandomAgent(),
#     "player_3": mjx.agent.RandomAgent(),
# }
# _mjx.EnvRunner.run(agents)
