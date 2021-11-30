import random
import threading
import time
from queue import Queue
from typing import Dict, List, Optional, Tuple

import _mjx  # type: ignore
from flask import Flask, redirect, render_template, request, url_for
from flask.views import MethodView, View

import mjxproto
from mjx.action import Action
from mjx.const import ActionType
from mjx.env import MjxEnv
from mjx.observation import Observation
from mjx.visualizer.converter import action_type_ja, get_tile_char
from mjx.visualizer.open_utils import open_tile_ids
from mjx.visualizer.selector import Selector
from mjx.visualizer.svg import to_svg


class Agent(_mjx.Agent):  # type: ignore
    def __init__(self) -> None:
        _mjx.Agent.__init__(self)  # type: ignore

    def act(self, observation: Observation) -> Action:
        raise NotImplementedError

    def act_batch(self, observations: List[Observation]) -> List[Action]:
        return [self.act(obs) for obs in observations]

    def serve(
        self,
        socket_address: str,
        batch_size: int = 64,
        wait_limit_ms: int = 100,
        sleep_ms: int = 10,
    ):
        _mjx.AgentServer(self, socket_address, batch_size, wait_limit_ms, sleep_ms)  # type: ignore

    def _act(self, observation: _mjx.Observation) -> _mjx.Action:  # type: ignore
        return self.act(Observation._from_cpp_obj(observation))._cpp_obj

    def _act_batch(self, observations: List[_mjx.Observation]) -> List[_mjx.Action]:  # type: ignore
        actions: List[Action] = self.act_batch(
            [Observation._from_cpp_obj(obs) for obs in observations]
        )
        return [action._cpp_obj for action in actions]


class RandomAgent(Agent):
    def __init__(self) -> None:
        super().__init__()

    def act(self, observation: Observation) -> Action:
        return random.choice(observation.legal_actions())


class TsumogiriAgent(Agent):
    def __init__(self) -> None:
        super().__init__()

    def act(self, observation: Observation) -> Action:
        legal_actions = observation.legal_actions()
        if len(legal_actions) == 1:
            return legal_actions[0]
        for action in legal_actions:
            if action.type() in [ActionType.TSUMOGIRI, ActionType.PASS]:
                return action
        assert False


class ShantenAgent(Agent):
    """A rule-based agent, which plays just to reduce the shanten-number.
    The logic is basically intended to reproduce Mjai's ShantenPlayer.

    - Mjai https://github.com/gimite/mjai
    """

    def __init__(self) -> None:
        super().__init__()

    def act(self, observation: Observation) -> Action:
        legal_actions = observation.legal_actions()
        if len(legal_actions) == 1:
            return legal_actions[0]

        # if it can win, just win
        win_actions = [a for a in legal_actions if a.type() in [ActionType.TSUMO, ActionType.RON]]
        if len(win_actions) >= 1:
            assert len(win_actions) == 1
            return win_actions[0]

        # if it can declare riichi, just declar riichi
        riichi_actions = [a for a in legal_actions if a.type() == ActionType.RIICHI]
        if len(riichi_actions) >= 1:
            assert len(riichi_actions) == 1
            return riichi_actions[0]

        # if it can apply chi/pon/open-kan, choose randomly
        steal_actions = [
            a
            for a in legal_actions
            if a.type() in [ActionType.CHI, ActionType.PON, ActionType, ActionType.OPEN_KAN]
        ]
        if len(steal_actions) >= 1:
            return random.choice(steal_actions)

        # if it can apply closed-kan/added-kan, choose randomly
        kan_actions = [
            a for a in legal_actions if a.type() in [ActionType.CLOSED_KAN, ActionType.ADDED_KAN]
        ]
        if len(kan_actions) >= 1:
            return random.choice(kan_actions)

        # discard an effective tile randomly
        legal_discards = [
            a for a in legal_actions if a.type() in [ActionType.DISCARD, ActionType.TSUMOGIRI]
        ]
        effective_discard_types = observation.curr_hand().effective_discard_types()
        effective_discards = [
            a for a in legal_discards if a.tile().type() in effective_discard_types
        ]
        if len(effective_discards) > 0:
            return random.choice(effective_discards)

        # if no effective tile exists, discard randomly
        return random.choice(legal_discards)


class RandomDebugAgent(Agent):
    def __init__(self) -> None:
        super().__init__()
        self._agent = _mjx.RandomDebugAgent()  # type: ignore

    def act(self, observation: Observation) -> Action:
        return Action._from_cpp_obj(self._act(observation._cpp_obj))

    def _act(self, observation: _mjx.Observation) -> _mjx.Action:  # type: ignore
        return self._agent._act(observation)


class RuleBasedAgent(Agent):
    def __init__(self) -> None:
        super().__init__()
        self._agent = _mjx.RuleBasedAgent()  # type: ignore

    def act(self, observation: Observation) -> Action:
        return Action._from_cpp_obj(self._act(observation._cpp_obj))

    def _act(self, observation: _mjx.Observation) -> _mjx.Action:  # type: ignore
        return self._agent._act(observation)


class HumanControlAgent(Agent):  # type: ignore
    def __init__(self, unicode: bool = False, rich: bool = False, ja: bool = False) -> None:
        super().__init__()
        self.unicode: bool = unicode
        self.ja: bool = ja
        self.rich: bool = rich

    def act(self, observation: Observation) -> Action:  # type: ignore
        return Selector.select_from_proto(
            observation.to_proto(), unicode=self.unicode, rich=self.rich, ja=self.ja
        )


class PostPage(MethodView):
    def get(self):
        return redirect(url_for("show"))

    def post(self):
        action_idx = request.form.get("choice")
        assert action_idx is not None
        assert ShowPage.observation is not None
        action = ShowPage.observation.legal_actions()[int(action_idx)]
        ShowPage.q.put(action)
        ShowPage.q.join()
        ShowPage.observation = ShowPage.q.get(block=True, timeout=None)
        ShowPage.q.task_done()
        return redirect(url_for("show"))


class ShowPage(View):
    q: Queue
    methods = ["GET", "POST"]
    observation: Optional[Observation] = None

    def dispatch_request(self):
        if ShowPage.observation is None:
            ShowPage.observation = ShowPage.q.get(block=True, timeout=None)
            ShowPage.q.task_done()

        svg_str = to_svg(ShowPage.observation.to_json())
        choices = ShowPage.make_choices(ShowPage.observation)
        return render_template("index.html", svg=svg_str, choices=choices)

    @classmethod
    def make_choices(cls, observation):
        assert observation is not None
        legal_actions_proto_ = [
            (action.to_proto(), i) for i, action in enumerate(observation.legal_actions())
        ]
        legal_actions_proto = cls.sort_legal_actions(legal_actions_proto_)
        choices = [cls.make_choice(choice_[0], choice_[1]) for choice_ in legal_actions_proto]
        return choices

    @classmethod
    def make_choice(cls, action_proto: mjxproto.Action, i: int):
        red_hai = [16, 52, 88]
        if action_proto.type == mjxproto.ActionType.ACTION_TYPE_DUMMY:
            return {"text": "<span>次へ</span>", "index": i}
        elif action_proto.type == mjxproto.ActionType.ACTION_TYPE_NO:
            return {"text": "<span>パス</span>", "index": i}
        elif action_proto.type in [
            mjxproto.ActionType.ACTION_TYPE_PON,
            mjxproto.ActionType.ACTION_TYPE_CHI,
            mjxproto.ActionType.ACTION_TYPE_CLOSED_KAN,
            mjxproto.ActionType.ACTION_TYPE_OPEN_KAN,
            mjxproto.ActionType.ACTION_TYPE_ADDED_KAN,
            mjxproto.ActionType.ACTION_TYPE_RON,
        ]:
            text = "<span>" + action_type_ja[action_proto.type] + "</span>"
            open_ids = [id for id in open_tile_ids(action_proto.open)]
            for hai_id in open_ids:
                if hai_id in red_hai:
                    text_ = '<span class="MJFontR">' + get_tile_char(hai_id, True) + "</span>"
                    text += text_
                else:
                    text_ = '<span class="MJFont">' + get_tile_char(hai_id, True) + "</span>"
                    text += text_
            return {"text": text, "index": i}
        else:
            text = "<span>" + action_type_ja[action_proto.type] + "</span>"
            if action_proto.tile in red_hai:
                text_ = (
                    '<span class="MJFontR">' + get_tile_char(action_proto.tile, True) + "</span>"
                )
                text += text_
            else:
                text_ = (
                    '<span class="MJFont">' + get_tile_char(action_proto.tile, True) + "</span>"
                )
                text += text_
            return {"text": text, "index": i}

    @classmethod
    def sort_legal_actions(cls, legal_actions_proto: List[Tuple[mjxproto.Action, int]]):
        """
        ツモ切りを先頭に持ってくるための専用ソート
        """
        legal_actions_proto_ = sorted(
            [a for a in legal_actions_proto if a[0].type != mjxproto.ACTION_TYPE_TSUMOGIRI],
            key=lambda x: x[0].type,
        )
        for action_ in legal_actions_proto:
            if action_[0].type == mjxproto.ACTION_TYPE_TSUMOGIRI:
                legal_actions_proto_ = [action_] + legal_actions_proto_
        return legal_actions_proto_


class HumanControlAgentOnBrowser(Agent):  # type: ignore
    def __init__(self) -> None:
        super().__init__()
        self.q = Queue()

        self.sub = threading.Thread(target=self.flask, args=(self.q,))
        self.sub.setDaemon(True)
        self.sub.start()
        time.sleep(2.0)

    def act(self, observation: Observation) -> Action:
        assert isinstance(observation, Observation)
        print("YOUR TURN")
        self.q.put(observation)
        self.q.join()

        action = self.q.get(block=True, timeout=None)
        self.q.task_done()
        return action

    def flask(self, q):
        page1 = ShowPage()
        page2 = PostPage()
        ShowPage.q = q
        app = Flask(__name__)
        app.config["SECRET_KEY"] = "8888"
        app.add_url_rule("/", view_func=page1.as_view("show"))
        app.add_url_rule("/post/", view_func=page2.as_view("post"))

        app.run()


def validate_agent(agent: Agent, n_games=1, use_batch=False):
    env = MjxEnv()
    for i in range(n_games):
        obs_dict: Dict[str, Observation] = env.reset()
        while not env.done():
            action_dict: Dict[str, Action] = {}
            for player_id, obs in obs_dict.items():
                action = agent.act_batch([obs])[0] if use_batch else agent.act(obs)
                assert action in obs.legal_actions()
                action_dict[player_id] = action
            obs_dict = env.step(action_dict)
