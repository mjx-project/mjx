import mjx

agents = {
    "player_0": "127.0.0.1:9091",
    "player_1": "127.0.0.1:9090",
    "player_2": "127.0.0.1:9090",
    "player_3": "127.0.0.1:9090",
}
mjx.run(agents, 1, 1, 1)
