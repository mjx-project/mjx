from google.protobuf import json_format

import mjx
import mjxproto
from mjx.const import ActionType, PlayerIdx

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
        mjx.Action('{"tile":10}'),
        mjx.Action('{"tile":1}'),
    ]
    action = mjx.Action.select_from(0, legal_actions)
    assert action.to_json() == '{"tile":1}'


def test_from_proto():
    action = mjx.Action(json_str)
    proto_action = action.to_proto()
    mjx.Action.from_proto(proto_action).to_json() == json_str


def test_type():
    action = mjx.Action(json_str)
    assert action.type() == 0
    assert action.type() == ActionType.DISCARD
    assert action.type() != 1
    assert action.type() != ActionType.TSUMOGIRI


def test_tile():
    action = mjx.Action('{"tile":10}')
    tile = action.tile()
    assert tile is not None
    assert tile.id() == 10

    action = mjx.Action('{"who":1,"type":"ACTION_TYPE_CHI","open":30111}')
    tile = action.tile()
    assert tile is None


def test_tile():
    action = mjx.Action('{"tile":10}')
    who = action.who()
    assert who == 0
    assert who == PlayerIdx.INIT_EAST
