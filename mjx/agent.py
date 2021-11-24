import random
import threading
import time
from queue import Queue
from typing import Dict, List, Optional, Tuple

import _mjx  # type: ignore
from flask import Flask, render_template, request
from flask.views import View

import mjxproto
from mjx.action import Action
from mjx.observation import Observation
from mjx.visualizer.converter import action_type_en, action_type_ja, get_tile_char
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


class RandomAgent(Agent):  # type: ignore
    def __init__(self) -> None:
        super().__init__()

    def act(self, observation: Observation) -> Action:  # type: ignore
        return random.choice(observation.legal_actions())


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


class ShowPages(View):
    q: Queue
    methods = ["GET", "POST"]
    observation: Optional[Observation] = None

    def dispatch_request(self):
        if False or request.method == "POST":
            assert ShowPages.observation is not None
            action_idx = request.form.get("choice")
            assert action_idx is not None

            action = ShowPages.observation.legal_actions()[int(action_idx)]
            ShowPages.q.put(action)
            ShowPages.q.join()

            ShowPages.observation = ShowPages.q.get(block=True, timeout=None)
            ShowPages.q.task_done()
            time.sleep(1.0)

            svg_str = to_svg(ShowPages.observation.to_json())
            choices = self.make_choices()
            return render_template("index.html", svg=svg_str, choices=choices)

        else:  # リクエストが非POSTとなるのは初回の表示のみという想定
            ShowPages.observation = ShowPages.q.get(block=True, timeout=None)
            ShowPages.q.task_done()
            time.sleep(1.0)

            svg_str = to_svg(ShowPages.observation.to_json())
            choices = self.make_choices()
            return render_template("index.html", svg=svg_str, choices=choices)

    def make_choices(self):
        assert ShowPages.observation is not None
        legal_actions_proto_ = [
            (action.to_proto(), i)
            for i, action in enumerate(ShowPages.observation.legal_actions())
        ]
        legal_actions_proto = self.sort_legal_actions(legal_actions_proto_)
        choices = [self.make_choice(choice_[0], choice_[1]) for choice_ in legal_actions_proto]
        return choices

    def make_choice(self, action_proto: mjxproto.Action, i: int):
        red_hai = [16, 52, 88]
        if action_proto.type == mjxproto.ActionType.ACTION_TYPE_DUMMY:
            return {"text": "<span>次へ</span>", "index": 0}
        elif action_proto.type == mjxproto.ActionType.ACTION_TYPE_NO:
            return {"text": "<span>パス</span>", "index": 0}
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

    def sort_legal_actions(self, legal_actions_proto: List[Tuple[mjxproto.Action, int]]):
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
        time.sleep(1.0)
        return action

    def flask(self, q):
        page = ShowPages()
        ShowPages.q = q
        app = Flask(__name__)
        app.add_url_rule("/", view_func=page.as_view("show"))
        app.run()
