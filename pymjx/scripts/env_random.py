import random

from mjx.env import RLlibMahjongEnv


def main():
    random.seed(2)
    env = RLlibMahjongEnv()
    env.seed(2)
    obs_dict = env.reset()
    dones = {"__all__": False}
    while not dones["__all__"]:
        act_dict = {id: random.choice(obs.legal_actions) for id, obs in obs_dict.items()}
        obs_dict, rewards, dones, info = env.step(act_dict)


if __name__ == "__main__":
    main()
