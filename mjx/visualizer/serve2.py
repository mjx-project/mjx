import mjx.agent

human_control_agent = mjx.agent.HumanControlAgent()
human_control_agent.serve("127.0.0.1:9091", 1, 0, 0)
