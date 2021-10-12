import mjx.agent

human_control_agent = mjx.agent.RandomAgent()
human_control_agent.serve("127.0.0.1:9091", 1, 0, 0)
