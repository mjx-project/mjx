import mjx.agent
import _mjx

agent = mjx.agent.RandomAgent()
agent.serve("127.0.0.1:9090", 1, 0, 0)
