import environment
import random


def main():
    random.seed(2)

    env = environment.MahjongEnv()
    env.seed(2)
    obs_dict = env.reset()

    for step in range(100):
        print(step)
        print(obs_dict)
        act_dict = {}
        for id, obs in obs_dict.items():
            act_dict[id] = random.choice(obs.legal_actions)
            print(act_dict[id])
        obs_dict, rewards, dones, info = env.step(act_dict)


if __name__ == '__main__':
    main()