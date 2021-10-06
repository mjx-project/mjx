import os

import mjx
from mjx import observation


def test_Observation():
    json_str = '{"publicObservation":{"playerIds":["player_2","player_1","player_0","player_3"],"initScore":{"tens":[25000,25000,25000,25000]},"doraIndicators":[101],"events":[{"type":"EVENT_TYPE_DRAW"}]},"privateObservation":{"initHand":{"closedTiles":[24,3,87,124,37,42,58,134,92,82,122,18,117]},"drawHistory":[79],"currHand":{"closedTiles":[3,18,24,37,42,58,79,82,87,92,117,122,124,134]}},"legalActions":[{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":3},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":18},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":24},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":37},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":42},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":58},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","type":"ACTION_TYPE_TSUMOGIRI","tile":79},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":82},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":87},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":92},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":117},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":122},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":124},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":134}]}'
    obs = mjx.Observation(json_str)
    restored = obs.to_json()
    assert json_str == restored


def test_legal_actions():
    json_str = '{"publicObservation":{"playerIds":["player_2","player_1","player_0","player_3"],"initScore":{"tens":[25000,25000,25000,25000]},"doraIndicators":[101],"events":[{"type":"EVENT_TYPE_DRAW"}]},"privateObservation":{"initHand":{"closedTiles":[24,3,87,124,37,42,58,134,92,82,122,18,117]},"drawHistory":[79],"currHand":{"closedTiles":[3,18,24,37,42,58,79,82,87,92,117,122,124,134]}},"legalActions":[{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":3},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":18},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":24},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":37},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":42},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":58},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","type":"ACTION_TYPE_TSUMOGIRI","tile":79},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":82},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":87},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":92},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":117},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":122},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":124},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":134}]}'
    obs = mjx.Observation(json_str)
    legal_actions = obs.legal_actions()
    assert len(legal_actions) == 14


def test_action_mask():
    json_str = '{"publicObservation":{"playerIds":["player_2","player_1","player_0","player_3"],"initScore":{"tens":[25000,25000,25000,25000]},"doraIndicators":[101],"events":[{"type":"EVENT_TYPE_DRAW"}]},"privateObservation":{"initHand":{"closedTiles":[24,3,87,124,37,42,58,134,92,82,122,18,117]},"drawHistory":[79],"currHand":{"closedTiles":[3,18,24,37,42,58,79,82,87,92,117,122,124,134]}},"legalActions":[{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":3},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":18},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":24},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":37},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":42},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":58},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","type":"ACTION_TYPE_TSUMOGIRI","tile":79},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":82},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":87},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":92},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":117},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":122},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":124},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":134}]}'
    obs = mjx.Observation(json_str)
    action_mask = obs.action_mask()
    assert action_mask.sum() == 14.0


def test_to_proto():
    json_str = '{"publicObservation":{"playerIds":["player_2","player_1","player_0","player_3"],"initScore":{"tens":[25000,25000,25000,25000]},"doraIndicators":[101],"events":[{"type":"EVENT_TYPE_DRAW"}]},"privateObservation":{"initHand":{"closedTiles":[24,3,87,124,37,42,58,134,92,82,122,18,117]},"drawHistory":[79],"currHand":{"closedTiles":[3,18,24,37,42,58,79,82,87,92,117,122,124,134]}},"legalActions":[{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":3},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":18},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":24},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":37},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":42},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":58},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","type":"ACTION_TYPE_TSUMOGIRI","tile":79},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":82},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":87},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":92},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":117},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":122},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":124},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":134}]}'
    obs = mjx.Observation(json_str)
    proto = obs.to_proto()
    assert len(proto.legal_actions) == 14


def test_to_json():
    json_str = '{"publicObservation":{"playerIds":["player_2","player_1","player_0","player_3"],"initScore":{"tens":[25000,25000,25000,25000]},"doraIndicators":[101],"events":[{"type":"EVENT_TYPE_DRAW"}]},"privateObservation":{"initHand":{"closedTiles":[24,3,87,124,37,42,58,134,92,82,122,18,117]},"drawHistory":[79],"currHand":{"closedTiles":[3,18,24,37,42,58,79,82,87,92,117,122,124,134]}},"legalActions":[{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":3},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":18},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":24},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":37},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":42},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":58},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","type":"ACTION_TYPE_TSUMOGIRI","tile":79},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":82},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":87},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":92},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":117},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":122},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":124},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":134}]}'
    obs = mjx.Observation(json_str)
    restored = obs.to_json()
    assert json_str == restored


def test_save_svg():
    json_str = '{"publicObservation":{"playerIds":["player_2","player_1","player_0","player_3"],"initScore":{"tens":[25000,25000,25000,25000]},"doraIndicators":[101],"events":[{"type":"EVENT_TYPE_DRAW"}]},"privateObservation":{"initHand":{"closedTiles":[24,3,87,124,37,42,58,134,92,82,122,18,117]},"drawHistory":[79],"currHand":{"closedTiles":[3,18,24,37,42,58,79,82,87,92,117,122,124,134]}},"legalActions":[{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":3},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":18},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":24},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":37},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":42},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":58},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","type":"ACTION_TYPE_TSUMOGIRI","tile":79},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":82},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":87},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":92},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":117},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":122},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":124},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":134}]}'
    obs = mjx.Observation(json_str)
    fname = "obs.svg"
    obs.save_svg(fname)
    os.remove(fname)


def test_from_proto():
    json_str = '{"publicObservation":{"playerIds":["player_2","player_1","player_0","player_3"],"initScore":{"tens":[25000,25000,25000,25000]},"doraIndicators":[101],"events":[{"type":"EVENT_TYPE_DRAW"}]},"privateObservation":{"initHand":{"closedTiles":[24,3,87,124,37,42,58,134,92,82,122,18,117]},"drawHistory":[79],"currHand":{"closedTiles":[3,18,24,37,42,58,79,82,87,92,117,122,124,134]}},"legalActions":[{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":3},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":18},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":24},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":37},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":42},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":58},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","type":"ACTION_TYPE_TSUMOGIRI","tile":79},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":82},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":87},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":92},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":117},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":122},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":124},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":134}]}'
    obs = mjx.Observation(json_str)
    proto_obs = obs.to_proto()
    mjx.Observation.from_proto(proto_obs).to_json() == json_str
