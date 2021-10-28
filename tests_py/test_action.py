from google.protobuf import json_format

import mjx
import mjxproto

json_str = '{"tile":10}'


def test_Action():
    action = mjx.Action(json_str)
    restored = action.to_json()
    assert json_str == restored


def test_to_json():
    action = mjx.Action(json_str)
    restored = action.to_json()
    assert json_str == restored


def test_to_proto():
    action = mjx.Action(json_str)
    action.to_proto() == json_format.Parse(json_str, mjxproto.Action())


def test_to_idx():
    action = mjx.Action(json_str)
    idx = action.to_idx()
    assert idx == 2


def test_select_from():
    legal_actions = [
        mjx.Action('{"tile":0}'),
        mjx.Action('{"tile":10}'),
    ]
    action = mjx.Action.select_from(0, legal_actions)
    action.to_json() == '{"tile":0}'


def test_from_proto():
    action = mjx.Action(json_str)
    proto_action = action.to_proto()
    mjx.Action.from_proto(proto_action).to_json() == json_str
