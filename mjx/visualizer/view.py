import time
from queue import Queue
from typing import List, Optional, Tuple

from flask import render_template, request
from flask.views import View

import mjxproto
from mjx.observation import Observation
from mjx.visualizer.converter import action_type_ja, get_tile_char
from mjx.visualizer.open_utils import open_tile_ids
from mjx.visualizer.svg import to_svg


class ShowPage(View):
    q: Queue
    methods = ["GET", "POST"]
    observation: Optional[Observation] = None

    def dispatch_request(self):
        if request.method == "POST":
            assert ShowPage.observation is not None
            action_idx = request.form.get("choice")
            assert action_idx is not None
            action = ShowPage.observation.legal_actions()[int(action_idx)]
            ShowPage.q.put(action)
            ShowPage.q.join()
            time.sleep(0.1)

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
