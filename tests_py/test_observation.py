import os

import mjx

json_str = '{"publicObservation":{"playerIds":["player_2","player_1","player_0","player_3"],"initScore":{"tens":[25000,25000,25000,25000]},"doraIndicators":[101],"events":[{"type":"EVENT_TYPE_DRAW"}]},"privateObservation":{"initHand":{"closedTiles":[24,3,87,124,37,42,58,134,92,82,122,18,117]},"drawHistory":[79],"currHand":{"closedTiles":[3,18,24,37,42,58,79,82,87,92,117,122,124,134]}},"legalActions":[{"tile":3},{"tile":18},{"tile":24},{"tile":37},{"tile":42},{"tile":58},{"type":"ACTION_TYPE_TSUMOGIRI","tile":79},{"tile":82},{"tile":87},{"tile":92},{"tile":117},{"tile":122},{"tile":124},{"tile":134}]}'


def test_Observation():
    obs = mjx.Observation(json_str)
    restored = obs.to_json()
    assert json_str == restored


def test_legal_actions():
    obs = mjx.Observation(json_str)
    legal_actions = obs.legal_actions()
    assert len(legal_actions) == 14


def test_action_mask():
    obs = mjx.Observation(json_str)
    action_mask = obs.action_mask()
    assert action_mask.sum() == 14.0


def test_to_proto():
    obs = mjx.Observation(json_str)
    proto = obs.to_proto()
    assert len(proto.legal_actions) == 14


def test_to_json():
    obs = mjx.Observation(json_str)
    restored = obs.to_json()
    assert json_str == restored


def test_save_svg():
    obs = mjx.Observation(json_str)
    fname = "obs.svg"
    obs.save_svg(fname)
    os.remove(fname)


def test_from_proto():
    obs = mjx.Observation(json_str)
    proto_obs = obs.to_proto()
    mjx.Observation.from_proto(proto_obs).to_json() == json_str


def test_add_legal_actions():
    json_wo_legal_actions = '{"publicObservation":{"playerIds":["player_2","player_1","player_0","player_3"],"initScore":{"tens":[25000,25000,25000,25000]},"doraIndicators":[101],"events":[{"type":"EVENT_TYPE_DRAW"}]},"privateObservation":{"initHand":{"closedTiles":[24,3,87,124,37,42,58,134,92,82,122,18,117]},"drawHistory":[79],"currHand":{"closedTiles":[3,18,24,37,42,58,79,82,87,92,117,122,124,134]}}}'
    json_restored = mjx.Observation.add_legal_actions(json_wo_legal_actions)
    assert json_str == json_restored


def test_mjx_large_v0():
    obs = mjx.Observation(json_str)
    rows = sum(
        [
            7,  # currentHand
            2,  # targetTile
            4,  # underRiichis
            12,  # discardedTiles
            12,  # discardedFromHand
            28,  # openedTiles
            4,  # dealer
            4,  # doras
            7,  # shanten
            1,  # effectiveDiscards
            1,  # effectiveDraws
            4,  # ignoredTiles
            5,  # kyotaku
            12,  # rankings
            7,  # round
            5,  # honba
            4,  # doraNumOfTarget
            13,  # doraNumInHand
        ]
    )
    assert obs.to_features("mjx-large-v0").shape == (rows, 34)
