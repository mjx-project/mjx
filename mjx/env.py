import os
from typing import Dict, List, Optional

import _mjx  # type: ignore

from mjx.action import Action
from mjx.observation import Observation
from mjx.state import State


class MjxEnv:
    def __init__(
        self,
        player_ids: List[str] = ["player_0", "player_1", "player_2", "player_3"],
    ):
        self._env: _mjx.MjxEnv = _mjx.MjxEnv(player_ids)  # type: ignore

    def reset(
        self, seed: Optional[int] = None, dealer_order: Optional[List[str]] = None
    ) -> Dict[str, Observation]:
        assert seed is None or seed >= 0
        cpp_obs_dict: Dict[str, _mjx.Observation] = self._env.reset(seed, dealer_order)  # type: ignore
        return {k: Observation._from_cpp_obj(v) for k, v in cpp_obs_dict.items()}

    def step(self, aciton_dict: Dict[str, Action]) -> Dict[str, Observation]:
        cpp_action_dict: Dict[str, _mjx.Action] = {k: v._cpp_obj for k, v in aciton_dict.items()}  # type: ignore
        cpp_obs_dict: Dict[str, _mjx.Observation] = self._env.step(cpp_action_dict)  # type: ignore
        return {k: Observation._from_cpp_obj(v) for k, v in cpp_obs_dict.items()}

    def done(self, done_type: str = "game") -> bool:
        """When done() returns true, the corresponding observations and rewards have
        the terminal information of the game (or round).
        """
        assert done_type in (
            "game",
            "round",
        ), f'Wrong done_type: "{done_type}".'
        return self._env.done(done_type)

    def rewards(self, reward_type: str = "game_tenhou_7dan") -> Dict[str, int]:
        assert reward_type in (
            "game_tenhou_7dan",
            "round_win",
        ), f'Wrong reward_type: "{reward_type}".'
        return self._env.rewards(reward_type)

    def state(self) -> State:
        return State._from_cpp_obj(self._env.state())


def run(
    agent_addresses: Dict[str, str],
    num_games: int,
    num_parallels: Optional[int] = None,  # default = # cpus
    show_interval: int = 100,
    states_save_dir: Optional[str] = None,
    seed_type: str = "random",
):
    if num_parallels is None:
        import multiprocessing

        num_parallels = multiprocessing.cpu_count()
    assert len(agent_addresses) == 4
    assert num_games >= 1
    assert num_parallels >= 1
    if states_save_dir:
        assert os.path.isdir(states_save_dir)

    agents = {k: _mjx.GrpcAgent(addr) for k, addr in agent_addresses.items()}  # type: ignore

    # define seed geenrators
    if seed_type == "random":
        seed_generator = _mjx.RandomSeedGenerator(list(agent_addresses.keys()))  # type: ignore
    elif seed_type == "duplicate":
        seed_generator = _mjx.DuplicateRandomSeedGenerator(list(agent_addresses.keys()))  # type: ignore
    else:
        assert False, f"Wrong seed_type: {seed_type}"

    results_save_file: Optional[str] = None  # TODO: fix
    _mjx.EnvRunner(agents, seed_generator, num_games, num_parallels, show_interval, states_save_dir, results_save_file)  # type: ignore
