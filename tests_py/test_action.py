from google.protobuf import json_format

import mjx
import mjxproto


def test_Action():
    json_str = '{"gameId":"xxx","tile":3}'
    action = mjx.Action(json_str)
    restored = action.to_json()
    assert json_str == restored


def test_to_json():
    json_str = '{"gameId":"xxx","tile":3}'
    action = mjx.Action(json_str)
    restored = action.to_json()
    assert json_str == restored


def test_to_proto():
    json_str = '{"gameId":"xxx","tile":10}'
    action = mjx.Action(json_str)
    action.to_proto() == json_format.Parse(json_str, mjxproto.Action())


def test_to_idx():
    json_str = '{"gameId":"xxx","tile":10}'
    action = mjx.Action(json_str)
    idx = action.to_idx()
    assert idx == 2


def test_select_from():
    legal_actions = [
        mjx.Action('{"gameId":"xxx","tile":0}'),
        mjx.Action('{"gameId":"xxx","tile":10}'),
    ]
    action = mjx.Action.select_from(0, legal_actions)
    action.to_json() == '{"gameId":"xxx","tile":0}'


def test_from_proto():
    json_str = '{"gameId":"xxx","tile":10}'
    action = mjx.Action(json_str)
    proto_action = action.to_proto()
    mjx.Action.from_proto(proto_action).to_json() == json_str
