import mjx

host = "127.0.0.1:8080"

mjx.run(
    {f"player_{i}": host for i in range(4)},
    num_games=1000,
    num_parallels=32
)
