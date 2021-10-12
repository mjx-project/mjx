import mjx

"""
このテストではカーソルキーを利用して選択する動作を行いますが、
pytestで実行するのが難しいため、わざとファイル名を変えて自動テストされないようにしています。

動作を確認するときは、通常のpythonファイルと同様に
python _test_run.py
としてください。
"""


def run_test():
    agents = {
        "player_0": "127.0.0.1:9091",
        "player_1": "127.0.0.1:9090",
        "player_2": "127.0.0.1:9090",
        "player_3": "127.0.0.1:9090",
    }
    mjx.run(agents, 1, 1, 1)


if __name__ == "__main__":
    run_test()
