import random

from mjx.environment import MahjongEnv


def main():
    random.seed(2)
    env = MahjongEnv()
    env.seed(2)
    obs_dict = env.reset()
    dones = {"__all__": False}
    while not dones["__all__"]:
        act_dict = {id: random.choice(obs.legal_actions) for id, obs in obs_dict.items()}
        obs_dict, rewards, dones, info = env.step(act_dict)


if __name__ == "__main__":
    main()
