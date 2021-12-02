from google.protobuf import json_format

import mjx
import mjxproto
from mjx.const import ActionType


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


def test_type():
    json_str = '{"gameId":"xxx","tile":10}'
    action = mjx.Action(json_str)
    assert action.type() == 0
    assert action.type() == ActionType.DISCARD
    assert action.type() != 1
    assert action.type() != ActionType.TSUMOGIRI


def test_tile():
    json_str = '{"gameId":"xxx","tile":10}'
    action = mjx.Action(json_str)
    tile = action.tile()
    assert tile is not None
    assert tile.id() == 10

    json_str = '{"gameId":"xxx","who":1,"type":"ACTION_TYPE_CHI","open":30111}'
    action = mjx.Action(json_str)
    tile = action.tile()
    assert tile is None


def test_select_from():
    json_str = '{"publicObservation":{"playerIds":["player_2","player_1","player_0","player_3"],"initScore":{"tens":[25000,25000,25000,25000]},"doraIndicators":[101],"events":[{"type":"EVENT_TYPE_DRAW"}]},"privateObservation":{"initHand":{"closedTiles":[24,3,87,124,37,42,58,134,92,82,122,18,117]},"drawHistory":[79],"currHand":{"closedTiles":[3,18,24,37,42,58,79,82,87,92,117,122,124,134]}},"legalActions":[{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":3},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":18},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":24},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":37},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":42},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":58},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","type":"ACTION_TYPE_TSUMOGIRI","tile":79},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":82},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":87},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":92},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":117},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":122},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":124},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":134}]}'
    obs = mjx.Observation(json_str)
    legal_actions = obs.legal_actions()
    idx = legal_actions[0].to_idx()
    a = mjx.Action.select_from(idx, legal_actions)
    assert type(a) == mjx.Action
    assert a == legal_actions[0]
