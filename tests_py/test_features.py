import mjxproto
from mjx import Observation, State


def state2obs(state_json: str, who: int) -> Observation:
    state = State(state_json)
    state_proto = state.to_proto()
    observation_proto = mjxproto.Observation()
    # who
    observation_proto.who = who

    # public
    observation_proto.public_observation.CopyFrom(state_proto.public_observation)

    # private
    observation_proto.private_observation.CopyFrom(state_proto.private_observations[who])

    # round_terminal
    observation_proto.round_terminal.CopyFrom(state_proto.round_terminal)

    observation = Observation(
        Observation.add_legal_actions(Observation.from_proto(observation_proto).to_json())
    )
    return observation


def test_self_wind():
    json_str = '{"publicObservation":{"playerIds":["player_2","player_1","player_0","player_3"],"initScore":{"tens":[25000,25000,25000,25000]},"doraIndicators":[101],"events":[{"type":"EVENT_TYPE_DRAW"}]},"privateObservation":{"initHand":{"closedTiles":[24,3,87,124,37,42,58,134,92,82,122,18,117]},"drawHistory":[79],"currHand":{"closedTiles":[3,18,24,37,42,58,79,82,87,92,117,122,124,134]}},"legalActions":[{"tile":3},{"tile":18},{"tile":24},{"tile":37},{"tile":42},{"tile":58},{"type":"ACTION_TYPE_TSUMOGIRI","tile":79},{"tile":82},{"tile":87},{"tile":92},{"tile":117},{"tile":122},{"tile":124},{"tile":134}]}'
    obs = Observation(json_str)
    feature = obs.to_features("han22-v0")

    # index78: If t is wind of self
    # 27:East Wind, 28:South Wind
    assert feature[78][27]
    assert not feature[78][28]


def test_table_wind():
    json_str = '{"publicObservation":{"playerIds":["player_2","player_1","player_0","player_3"],"initScore":{"tens":[25000,25000,25000,25000]},"doraIndicators":[101],"events":[{"type":"EVENT_TYPE_DRAW"}]},"privateObservation":{"initHand":{"closedTiles":[24,3,87,124,37,42,58,134,92,82,122,18,117]},"drawHistory":[79],"currHand":{"closedTiles":[3,18,24,37,42,58,79,82,87,92,117,122,124,134]}},"legalActions":[{"tile":3},{"tile":18},{"tile":24},{"tile":37},{"tile":42},{"tile":58},{"type":"ACTION_TYPE_TSUMOGIRI","tile":79},{"tile":82},{"tile":87},{"tile":92},{"tile":117},{"tile":122},{"tile":124},{"tile":134}]}'
    obs = Observation(json_str)
    feature = obs.to_features("han22-v0")

    # index79: If t is wind of the table
    # 27: East Wind, 28:South Wind
    assert feature[79][27]
    assert not feature[79][28]


def test_dora_indicator():
    json_str = '{"who":3,"publicObservation":{"playerIds":["player_2","player_1","player_3","player_0"],"initScore":{"round":7,"honba":7,"tens":[31000,23000,23000,23000]},"doraIndicators":[106],"events":[{"type":"EVENT_TYPE_DRAW","who":3}]},"privateObservation":{"who":3,"initHand":{"closedTiles":[95,77,74,4,85,70,30,66,31,59,102,84,78]},"drawHistory":[37],"currHand":{"closedTiles":[4,30,31,37,59,66,70,74,77,78,84,85,95,102]}},"legalActions":[{"who":3,"tile":4},{"who":3,"tile":30},{"type":"ACTION_TYPE_TSUMOGIRI","who":3,"tile":37},{"who":3,"tile":59},{"who":3,"tile":66},{"who":3,"tile":70},{"who":3,"tile":74},{"who":3,"tile":77},{"who":3,"tile":84},{"who":3,"tile":95},{"who":3,"tile":102}]}'
    obs = Observation(json_str)
    feature = obs.to_features("han22-v0")

    # index70: If t is Dora indicator
    # 26: 9s
    assert feature[70][26]

    json_str = '{"who":1,"publicObservation":{"playerIds":["player_2","player_3","player_1","player_0"],"initScore":{"tens":[25000,25000,25000,25000]},"doraIndicators":[49,48],"events":[{"type":"EVENT_TYPE_DRAW"},{"tile":121},{"type":"EVENT_TYPE_DRAW","who":1},{"who":1,"tile":86},{"type":"EVENT_TYPE_CHI","who":2,"open":52471},{"who":2,"tile":69},{"type":"EVENT_TYPE_DRAW","who":3},{"who":3,"tile":47},{"type":"EVENT_TYPE_CHI","open":26103},{"tile":99},{"type":"EVENT_TYPE_DRAW","who":1},{"type":"EVENT_TYPE_TSUMOGIRI","who":1,"tile":104},{"type":"EVENT_TYPE_CHI","who":2,"open":63575},{"who":2,"tile":35},{"type":"EVENT_TYPE_PON","who":1,"open":13353},{"who":1,"tile":79},{"type":"EVENT_TYPE_DRAW","who":2},{"who":2,"tile":82},{"type":"EVENT_TYPE_DRAW","who":3},{"who":3,"tile":25},{"type":"EVENT_TYPE_DRAW"},{"type":"EVENT_TYPE_TSUMOGIRI","tile":17},{"type":"EVENT_TYPE_DRAW","who":1},{"who":1,"tile":83},{"type":"EVENT_TYPE_DRAW","who":2},{"who":2,"tile":12},{"type":"EVENT_TYPE_DRAW","who":3},{"who":3,"tile":29},{"type":"EVENT_TYPE_DRAW"},{"tile":6},{"type":"EVENT_TYPE_DRAW","who":1},{"who":1,"tile":36},{"type":"EVENT_TYPE_DRAW","who":2},{"who":2,"tile":45},{"type":"EVENT_TYPE_DRAW","who":3},{"who":3,"tile":114},{"type":"EVENT_TYPE_DRAW"},{"tile":14},{"type":"EVENT_TYPE_DRAW","who":1},{"type":"EVENT_TYPE_TSUMOGIRI","who":1,"tile":55},{"type":"EVENT_TYPE_DRAW","who":2},{"who":2,"tile":95},{"type":"EVENT_TYPE_DRAW","who":3},{"who":3,"tile":131},{"type":"EVENT_TYPE_DRAW"},{"tile":123},{"type":"EVENT_TYPE_DRAW","who":1},{"who":1,"tile":58},{"type":"EVENT_TYPE_DRAW","who":2},{"type":"EVENT_TYPE_TSUMOGIRI","who":2,"tile":60},{"type":"EVENT_TYPE_PON","who":3,"open":23147},{"who":3,"tile":96},{"type":"EVENT_TYPE_DRAW"},{"tile":67},{"type":"EVENT_TYPE_DRAW","who":1},{"who":1,"tile":134},{"type":"EVENT_TYPE_DRAW","who":2},{"who":2,"tile":13},{"type":"EVENT_TYPE_DRAW","who":3},{"who":3,"tile":1},{"type":"EVENT_TYPE_DRAW"},{"tile":88},{"type":"EVENT_TYPE_DRAW","who":1},{"who":1,"tile":5},{"type":"EVENT_TYPE_DRAW","who":2},{"who":2,"tile":39},{"type":"EVENT_TYPE_DRAW","who":3},{"who":3},{"type":"EVENT_TYPE_DRAW"},{"tile":85},{"type":"EVENT_TYPE_DRAW","who":1},{"who":1,"tile":53},{"type":"EVENT_TYPE_PON","open":20585},{"tile":87},{"type":"EVENT_TYPE_DRAW","who":1},{"who":1,"tile":57},{"type":"EVENT_TYPE_DRAW","who":2},{"who":2,"tile":111},{"type":"EVENT_TYPE_DRAW","who":3},{"who":3,"tile":7},{"type":"EVENT_TYPE_DRAW"},{"tile":126},{"type":"EVENT_TYPE_DRAW","who":1},{"who":1,"tile":90},{"type":"EVENT_TYPE_DRAW","who":2},{"who":2,"tile":112},{"type":"EVENT_TYPE_DRAW","who":3},{"who":3,"tile":71},{"type":"EVENT_TYPE_DRAW"},{"tile":92},{"type":"EVENT_TYPE_DRAW","who":1},{"who":1,"tile":128},{"type":"EVENT_TYPE_DRAW","who":2},{"type":"EVENT_TYPE_TSUMOGIRI","who":2,"tile":110},{"type":"EVENT_TYPE_PON","who":3,"open":42603},{"who":3,"tile":94},{"type":"EVENT_TYPE_DRAW"},{"tile":116},{"type":"EVENT_TYPE_DRAW","who":1},{"who":1,"tile":27},{"type":"EVENT_TYPE_DRAW","who":2},{"who":2,"tile":132},{"type":"EVENT_TYPE_DRAW","who":3},{"type":"EVENT_TYPE_ADDED_KAN","who":3,"open":23155},{"type":"EVENT_TYPE_DRAW","who":3},{"type":"EVENT_TYPE_NEW_DORA","tile":48},{"who":3,"tile":11}]},"privateObservation":{"who":1,"initHand":{"closedTiles":[32,58,28,83,134,125,103,86,9,127,90,34,79]},"drawHistory":[36,104,128,8,55,5,53,57,27,18,80,50,72],"currHand":{"closedTiles":[8,9,18,28,50,72,80,103,125,127],"opens":[13353]}},"legalActions":[{"type":"ACTION_TYPE_PON","who":1,"open":4170},{"type":"ACTION_TYPE_NO","who":1}]}'
    obs = Observation(json_str)
    feature = obs.to_features("han22-v0")

    # index71: If t is Dora indicator (2 repeats)
    # 12: 4m
    assert feature[71][12]


def test_dora():
    json_str = '{"who":3,"publicObservation":{"playerIds":["player_2","player_1","player_3","player_0"],"initScore":{"round":7,"honba":7,"tens":[31000,23000,23000,23000]},"doraIndicators":[106],"events":[{"type":"EVENT_TYPE_DRAW","who":3}]},"privateObservation":{"who":3,"initHand":{"closedTiles":[95,77,74,4,85,70,30,66,31,59,102,84,78]},"drawHistory":[37],"currHand":{"closedTiles":[4,30,31,37,59,66,70,74,77,78,84,85,95,102]}},"legalActions":[{"who":3,"tile":4},{"who":3,"tile":30},{"type":"ACTION_TYPE_TSUMOGIRI","who":3,"tile":37},{"who":3,"tile":59},{"who":3,"tile":66},{"who":3,"tile":70},{"who":3,"tile":74},{"who":3,"tile":77},{"who":3,"tile":84},{"who":3,"tile":95},{"who":3,"tile":102}]}'
    obs = Observation(json_str)
    feature = obs.to_features("han22-v0")

    # index74: If t is Dora
    # 18: 1s
    assert feature[74][18]

    json_str = '{"who":1,"publicObservation":{"playerIds":["player_2","player_3","player_1","player_0"],"initScore":{"tens":[25000,25000,25000,25000]},"doraIndicators":[49,48],"events":[{"type":"EVENT_TYPE_DRAW"},{"tile":121},{"type":"EVENT_TYPE_DRAW","who":1},{"who":1,"tile":86},{"type":"EVENT_TYPE_CHI","who":2,"open":52471},{"who":2,"tile":69},{"type":"EVENT_TYPE_DRAW","who":3},{"who":3,"tile":47},{"type":"EVENT_TYPE_CHI","open":26103},{"tile":99},{"type":"EVENT_TYPE_DRAW","who":1},{"type":"EVENT_TYPE_TSUMOGIRI","who":1,"tile":104},{"type":"EVENT_TYPE_CHI","who":2,"open":63575},{"who":2,"tile":35},{"type":"EVENT_TYPE_PON","who":1,"open":13353},{"who":1,"tile":79},{"type":"EVENT_TYPE_DRAW","who":2},{"who":2,"tile":82},{"type":"EVENT_TYPE_DRAW","who":3},{"who":3,"tile":25},{"type":"EVENT_TYPE_DRAW"},{"type":"EVENT_TYPE_TSUMOGIRI","tile":17},{"type":"EVENT_TYPE_DRAW","who":1},{"who":1,"tile":83},{"type":"EVENT_TYPE_DRAW","who":2},{"who":2,"tile":12},{"type":"EVENT_TYPE_DRAW","who":3},{"who":3,"tile":29},{"type":"EVENT_TYPE_DRAW"},{"tile":6},{"type":"EVENT_TYPE_DRAW","who":1},{"who":1,"tile":36},{"type":"EVENT_TYPE_DRAW","who":2},{"who":2,"tile":45},{"type":"EVENT_TYPE_DRAW","who":3},{"who":3,"tile":114},{"type":"EVENT_TYPE_DRAW"},{"tile":14},{"type":"EVENT_TYPE_DRAW","who":1},{"type":"EVENT_TYPE_TSUMOGIRI","who":1,"tile":55},{"type":"EVENT_TYPE_DRAW","who":2},{"who":2,"tile":95},{"type":"EVENT_TYPE_DRAW","who":3},{"who":3,"tile":131},{"type":"EVENT_TYPE_DRAW"},{"tile":123},{"type":"EVENT_TYPE_DRAW","who":1},{"who":1,"tile":58},{"type":"EVENT_TYPE_DRAW","who":2},{"type":"EVENT_TYPE_TSUMOGIRI","who":2,"tile":60},{"type":"EVENT_TYPE_PON","who":3,"open":23147},{"who":3,"tile":96},{"type":"EVENT_TYPE_DRAW"},{"tile":67},{"type":"EVENT_TYPE_DRAW","who":1},{"who":1,"tile":134},{"type":"EVENT_TYPE_DRAW","who":2},{"who":2,"tile":13},{"type":"EVENT_TYPE_DRAW","who":3},{"who":3,"tile":1},{"type":"EVENT_TYPE_DRAW"},{"tile":88},{"type":"EVENT_TYPE_DRAW","who":1},{"who":1,"tile":5},{"type":"EVENT_TYPE_DRAW","who":2},{"who":2,"tile":39},{"type":"EVENT_TYPE_DRAW","who":3},{"who":3},{"type":"EVENT_TYPE_DRAW"},{"tile":85},{"type":"EVENT_TYPE_DRAW","who":1},{"who":1,"tile":53},{"type":"EVENT_TYPE_PON","open":20585},{"tile":87},{"type":"EVENT_TYPE_DRAW","who":1},{"who":1,"tile":57},{"type":"EVENT_TYPE_DRAW","who":2},{"who":2,"tile":111},{"type":"EVENT_TYPE_DRAW","who":3},{"who":3,"tile":7},{"type":"EVENT_TYPE_DRAW"},{"tile":126},{"type":"EVENT_TYPE_DRAW","who":1},{"who":1,"tile":90},{"type":"EVENT_TYPE_DRAW","who":2},{"who":2,"tile":112},{"type":"EVENT_TYPE_DRAW","who":3},{"who":3,"tile":71},{"type":"EVENT_TYPE_DRAW"},{"tile":92},{"type":"EVENT_TYPE_DRAW","who":1},{"who":1,"tile":128},{"type":"EVENT_TYPE_DRAW","who":2},{"type":"EVENT_TYPE_TSUMOGIRI","who":2,"tile":110},{"type":"EVENT_TYPE_PON","who":3,"open":42603},{"who":3,"tile":94},{"type":"EVENT_TYPE_DRAW"},{"tile":116},{"type":"EVENT_TYPE_DRAW","who":1},{"who":1,"tile":27},{"type":"EVENT_TYPE_DRAW","who":2},{"who":2,"tile":132},{"type":"EVENT_TYPE_DRAW","who":3},{"type":"EVENT_TYPE_ADDED_KAN","who":3,"open":23155},{"type":"EVENT_TYPE_DRAW","who":3},{"type":"EVENT_TYPE_NEW_DORA","tile":48},{"who":3,"tile":11}]},"privateObservation":{"who":1,"initHand":{"closedTiles":[32,58,28,83,134,125,103,86,9,127,90,34,79]},"drawHistory":[36,104,128,8,55,5,53,57,27,18,80,50,72],"currHand":{"closedTiles":[8,9,18,28,50,72,80,103,125,127],"opens":[13353]}},"legalActions":[{"type":"ACTION_TYPE_PON","who":1,"open":4170},{"type":"ACTION_TYPE_NO","who":1}]}'
    obs = Observation(json_str)
    feature = obs.to_features("han22-v0")

    # index75: If t is Dora (2 repeats)
    # 13: 5m
    assert feature[75][13]


def test_closed_tiles():
    json_str = '{"publicObservation":{"playerIds":["player_2","player_1","player_0","player_3"],"initScore":{"tens":[25000,25000,25000,25000]},"doraIndicators":[101],"events":[{"type":"EVENT_TYPE_DRAW"}]},"privateObservation":{"initHand":{"closedTiles":[24,3,87,124,37,42,58,134,92,82,122,18,117]},"drawHistory":[79],"currHand":{"closedTiles":[3,18,24,37,42,58,79,82,87,92,117,122,124,134]}},"legalActions":[{"tile":3},{"tile":18},{"tile":24},{"tile":37},{"tile":42},{"tile":58},{"type":"ACTION_TYPE_TSUMOGIRI","tile":79},{"tile":82},{"tile":87},{"tile":92},{"tile":117},{"tile":122},{"tile":124},{"tile":134}]}'
    obs = Observation(json_str)
    feature = obs.to_features("han22-v0")

    val = [
        True,
        False,
        False,
        False,
        True,
        False,
        True,
        False,
        False,
        True,
        True,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        False,
        True,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        False,
        True,
    ]

    # index0: If player 0 has >= 1 t in hand
    assert all(feature[0] == val)


def test_three_tiles_in_hand():
    json_str = '{"who":1,"publicObservation":{"playerIds":["player_1","player_2","player_0","player_3"],"initScore":{"tens":[25000,25000,25000,25000]},"doraIndicators":[6],"events":[{"type":"EVENT_TYPE_DRAW"},{"tile":27},{"type":"EVENT_TYPE_CHI","who":1,"open":16631},{"who":1,"tile":41},{"type":"EVENT_TYPE_DRAW","who":2},{"who":2,"tile":131},{"type":"EVENT_TYPE_DRAW","who":3},{"who":3,"tile":120},{"type":"EVENT_TYPE_DRAW"},{"tile":35},{"type":"EVENT_TYPE_DRAW","who":1},{"who":1,"tile":107},{"type":"EVENT_TYPE_DRAW","who":2},{"who":2,"tile":74},{"type":"EVENT_TYPE_DRAW","who":3},{"who":3,"tile":46},{"type":"EVENT_TYPE_DRAW"},{"tile":127},{"type":"EVENT_TYPE_DRAW","who":1},{"who":1,"tile":4},{"type":"EVENT_TYPE_DRAW","who":2},{"who":2,"tile":28},{"type":"EVENT_TYPE_DRAW","who":3},{"type":"EVENT_TYPE_TSUMOGIRI","who":3,"tile":1},{"type":"EVENT_TYPE_DRAW"},{"tile":75},{"type":"EVENT_TYPE_CHI","who":1,"open":43263},{"who":1,"tile":16},{"type":"EVENT_TYPE_DRAW","who":2},{"who":2,"tile":63},{"type":"EVENT_TYPE_DRAW","who":3},{"who":3,"tile":2},{"type":"EVENT_TYPE_DRAW"},{"tile":94},{"type":"EVENT_TYPE_DRAW","who":1},{"who":1,"tile":82},{"type":"EVENT_TYPE_DRAW","who":2},{"who":2,"tile":117},{"type":"EVENT_TYPE_DRAW","who":3},{"who":3,"tile":62},{"type":"EVENT_TYPE_DRAW"},{"tile":31},{"type":"EVENT_TYPE_DRAW","who":1},{"who":1,"tile":129},{"type":"EVENT_TYPE_DRAW","who":2},{"who":2,"tile":135},{"type":"EVENT_TYPE_DRAW","who":3},{"who":3,"tile":36},{"type":"EVENT_TYPE_DRAW"},{"tile":80},{"type":"EVENT_TYPE_DRAW","who":1},{"who":1,"tile":14},{"type":"EVENT_TYPE_DRAW","who":2},{"who":2,"tile":126},{"type":"EVENT_TYPE_DRAW","who":3},{"who":3,"tile":21},{"type":"EVENT_TYPE_DRAW"},{"tile":57},{"type":"EVENT_TYPE_DRAW","who":1},{"who":1,"tile":64},{"type":"EVENT_TYPE_DRAW","who":2},{"type":"EVENT_TYPE_TSUMOGIRI","who":2,"tile":130},{"type":"EVENT_TYPE_DRAW","who":3},{"who":3,"tile":47},{"type":"EVENT_TYPE_DRAW"},{"tile":110},{"type":"EVENT_TYPE_DRAW","who":1},{"who":1,"tile":66},{"type":"EVENT_TYPE_DRAW","who":2},{"type":"EVENT_TYPE_TSUMOGIRI","who":2,"tile":70},{"type":"EVENT_TYPE_DRAW","who":3},{"who":3,"tile":92},{"type":"EVENT_TYPE_DRAW"},{"tile":105},{"type":"EVENT_TYPE_DRAW","who":1}]},"privateObservation":{"who":1,"initHand":{"closedTiles":[81,107,79,82,41,99,4,14,22,97,29,66,129]},"drawHistory":[16,39,64,122,30,98,119,15],"currHand":{"closedTiles":[15,30,39,97,98,99,119,122],"opens":[16631,43263]}},"legalActions":[{"type":"ACTION_TYPE_TSUMOGIRI","who":1,"tile":15},{"who":1,"tile":30},{"who":1,"tile":39},{"who":1,"tile":97},{"who":1,"tile":119},{"who":1,"tile":122}]}'
    obs = Observation(json_str)
    feature = obs.to_features("han22-v0")

    # index2: If player 0 has >= 2 t in hand
    # 24: 2p
    assert feature[2][24]


def test_discarded_with_riichi():
    json_str = '{"who":1,"publicObservation":{"playerIds":["rule-based-0","target-player","rule-based-2","rule-based-3"],"initScore":{"round":2,"tens":[25200,24000,27300,23500]},"doraIndicators":[88],"events":[{"type":"EVENT_TYPE_DRAW","who":2},{"who":2,"tile":108},{"type":"EVENT_TYPE_DRAW","who":3},{"who":3,"tile":119},{"type":"EVENT_TYPE_DRAW"},{"tile":116},{"type":"EVENT_TYPE_DRAW","who":1},{"who":1,"tile":110},{"type":"EVENT_TYPE_DRAW","who":2},{"who":2,"tile":114},{"type":"EVENT_TYPE_DRAW","who":3},{"type":"EVENT_TYPE_TSUMOGIRI","who":3,"tile":111},{"type":"EVENT_TYPE_DRAW"},{"type":"EVENT_TYPE_TSUMOGIRI","tile":134},{"type":"EVENT_TYPE_DRAW","who":1},{"type":"EVENT_TYPE_TSUMOGIRI","who":1,"tile":122},{"type":"EVENT_TYPE_DRAW","who":2},{"who":2,"tile":39},{"type":"EVENT_TYPE_DRAW","who":3},{"who":3,"tile":84},{"type":"EVENT_TYPE_DRAW"},{"type":"EVENT_TYPE_TSUMOGIRI","tile":123},{"type":"EVENT_TYPE_DRAW","who":1},{"who":1,"tile":124},{"type":"EVENT_TYPE_DRAW","who":2},{"who":2,"tile":106},{"type":"EVENT_TYPE_DRAW","who":3},{"type":"EVENT_TYPE_TSUMOGIRI","who":3,"tile":25},{"type":"EVENT_TYPE_DRAW"},{"tile":37},{"type":"EVENT_TYPE_DRAW","who":1},{"type":"EVENT_TYPE_TSUMOGIRI","who":1,"tile":117},{"type":"EVENT_TYPE_DRAW","who":2},{"type":"EVENT_TYPE_TSUMOGIRI","who":2,"tile":36},{"type":"EVENT_TYPE_DRAW","who":3},{"type":"EVENT_TYPE_TSUMOGIRI","who":3,"tile":118},{"type":"EVENT_TYPE_DRAW"},{"tile":68},{"type":"EVENT_TYPE_DRAW","who":1},{"who":1,"tile":41},{"type":"EVENT_TYPE_DRAW","who":2},{"who":2},{"type":"EVENT_TYPE_DRAW","who":3},{"who":3,"tile":51},{"type":"EVENT_TYPE_PON","who":2,"open":19465},{"who":2,"tile":4},{"type":"EVENT_TYPE_DRAW","who":3},{"who":3,"tile":100},{"type":"EVENT_TYPE_CHI","open":60463},{"tile":42},{"type":"EVENT_TYPE_DRAW","who":1},{"who":1,"tile":69},{"type":"EVENT_TYPE_DRAW","who":2},{"type":"EVENT_TYPE_TSUMOGIRI","who":2,"tile":7},{"type":"EVENT_TYPE_PON","open":2570},{"tile":54},{"type":"EVENT_TYPE_DRAW","who":1},{"who":1,"tile":79},{"type":"EVENT_TYPE_DRAW","who":2},{"type":"EVENT_TYPE_TSUMOGIRI","who":2,"tile":120},{"type":"EVENT_TYPE_DRAW","who":3},{"type":"EVENT_TYPE_TSUMOGIRI","who":3,"tile":78},{"type":"EVENT_TYPE_DRAW"},{"type":"EVENT_TYPE_TSUMOGIRI","tile":1},{"type":"EVENT_TYPE_DRAW","who":1},{"who":1,"tile":85},{"type":"EVENT_TYPE_DRAW","who":2},{"who":2,"tile":43},{"type":"EVENT_TYPE_DRAW","who":3},{"type":"EVENT_TYPE_TSUMOGIRI","who":3,"tile":121},{"type":"EVENT_TYPE_DRAW"},{"tile":24},{"type":"EVENT_TYPE_DRAW","who":1},{"who":1,"tile":95},{"type":"EVENT_TYPE_DRAW","who":2},{"who":2,"tile":57},{"type":"EVENT_TYPE_DRAW","who":3},{"type":"EVENT_TYPE_RIICHI","who":3},{"who":3,"tile":59},{"type":"EVENT_TYPE_RIICHI_SCORE_CHANGE","who":3},{"type":"EVENT_TYPE_DRAW"},{"type":"EVENT_TYPE_TSUMOGIRI","tile":115},{"type":"EVENT_TYPE_DRAW","who":1},{"who":1,"tile":101},{"type":"EVENT_TYPE_DRAW","who":2},{"type":"EVENT_TYPE_TSUMOGIRI","who":2,"tile":70},{"type":"EVENT_TYPE_DRAW","who":3},{"type":"EVENT_TYPE_TSUMOGIRI","who":3,"tile":104},{"type":"EVENT_TYPE_DRAW"},{"type":"EVENT_TYPE_TSUMOGIRI","tile":112},{"type":"EVENT_TYPE_DRAW","who":1},{"type":"EVENT_TYPE_TSUMOGIRI","who":1,"tile":83},{"type":"EVENT_TYPE_PON","who":2,"open":31787},{"who":2,"tile":33},{"type":"EVENT_TYPE_DRAW","who":3},{"type":"EVENT_TYPE_TSUMO","who":3,"tile":67}]},"privateObservation":{"who":1,"initHand":{"closedTiles":[129,101,63,128,31,41,10,21,124,69,14,110,79]},"drawHistory":[26,122,95,117,55,85,58,13,61,44,83],"currHand":{"closedTiles":[10,13,14,21,26,31,44,55,58,61,63,128,129]}},"roundTerminal":{"finalScore":{"round":2,"tens":[22200,21000,21300,35500]},"wins":[{"who":3,"fromWho":3,"hand":{"closedTiles":[20,23,46,47,60,62,65,67,73,75,96,99,133,135]},"winTile":67,"fu":25,"ten":12000,"tenChanges":[-3000,-3000,-6000,13000],"yakus":[0,1,22,53],"fans":[1,1,2,2],"uraDoraIndicators":[16]}]},"legalActions":[{"who":1,"type":"ACTION_TYPE_DUMMY"}]}'
    obs = Observation(json_str)
    feature = obs.to_features("han22-v0")

    # index59: If player 1 has discarded t to announce Riichi
    # 14: 2p
    assert feature[59][14]


def test_can_riichi():
    json_str = '{"who":3,"publicObservation":{"playerIds":["player_2","player_0","player_1","player_3"],"initScore":{"round":7,"honba":7,"tens":[25000,25000,25000,25000]},"doraIndicators":[106],"events":[{"type":"EVENT_TYPE_DRAW","who":3},{"who":3,"tile":71},{"type":"EVENT_TYPE_DRAW"},{"tile":42},{"type":"EVENT_TYPE_DRAW","who":1},{"who":1,"tile":129},{"type":"EVENT_TYPE_DRAW","who":2},{"who":2,"tile":67},{"type":"EVENT_TYPE_DRAW","who":3},{"who":3,"tile":60},{"type":"EVENT_TYPE_PON","open":23147},{"tile":107},{"type":"EVENT_TYPE_DRAW","who":1},{"who":1,"tile":14},{"type":"EVENT_TYPE_DRAW","who":2},{"who":2,"tile":116},{"type":"EVENT_TYPE_DRAW","who":3},{"who":3,"tile":8},{"type":"EVENT_TYPE_CHI","open":2167},{"tile":29},{"type":"EVENT_TYPE_DRAW","who":1},{"who":1,"tile":64},{"type":"EVENT_TYPE_DRAW","who":2},{"who":2,"tile":124},{"type":"EVENT_TYPE_DRAW","who":3},{"who":3,"tile":80},{"type":"EVENT_TYPE_DRAW"},{"tile":66},{"type":"EVENT_TYPE_DRAW","who":1},{"who":1,"tile":41},{"type":"EVENT_TYPE_DRAW","who":2},{"who":2,"tile":32},{"type":"EVENT_TYPE_DRAW","who":3}]},"privateObservation":{"who":3,"initHand":{"closedTiles":[71,104,80,102,87,8,60,6,34,26,91,82,11]},"drawHistory":[78,23,79,15,18],"currHand":{"closedTiles":[6,11,15,18,23,26,34,78,79,82,87,91,102,104]}},"legalActions":[{"type":"ACTION_TYPE_RIICHI","who":3},{"who":3,"tile":6},{"who":3,"tile":11},{"who":3,"tile":15},{"type":"ACTION_TYPE_TSUMOGIRI","who":3,"tile":18},{"who":3,"tile":23},{"who":3,"tile":26},{"who":3,"tile":34},{"who":3,"tile":78},{"who":3,"tile":82},{"who":3,"tile":87},{"who":3,"tile":91},{"who":3,"tile":102},{"who":3,"tile":104}]}'
    obs = Observation(json_str)
    feature = obs.to_features("han22-v0")

    # index89: If Riichi is possible by discarding t
    # 8: 9m
    assert feature[89][8]


def test_can_chi():
    json_str = '{"who":3,"publicObservation":{"playerIds":["player_3","player_2","player_0","player_1"],"initScore":{"round":1,"honba":1,"tens":[25000,25000,25000,25000]},"doraIndicators":[6],"events":[{"type":"EVENT_TYPE_DRAW","who":1},{"who":1,"tile":71},{"type":"EVENT_TYPE_DRAW","who":2},{"who":2,"tile":84}]},"privateObservation":{"who":3,"initHand":{"closedTiles":[50,20,29,39,111,76,62,88,2,112,120,94,97]},"currHand":{"closedTiles":[2,20,29,39,50,62,76,88,94,97,111,112,120]}},"legalActions":[{"type":"ACTION_TYPE_CHI","who":3,"open":52487},{"type":"ACTION_TYPE_NO","who":3}]}'
    obs = Observation(json_str)
    feature = obs.to_features("han22-v0")

    # index82: If at t can be chi(smallest)
    # 21: 4s
    assert feature[82][21]


def test_kyuuhai():
    json_str = '{"hiddenState":{"wall":[2,75,111,69,33,114,64,113,30,4,92,55,73,36,21,1,128,11,115,57,25,90,108,50,80,96,132,26,10,17,60,87,51,119,47,35,83,3,41,107,13,86,133,37,103,6,82,116,123,120,40,52,101,18,61,23,77,67,27,131,135,31,0,112,29,121,93,78,104,19,38,79,24,12,105,68,118,65,66,22,70,34,91,122,97,46,130,39,71,43,129,127,32,9,14,16,98,88,48,58,85,54,81,15,102,7,42,53,76,117,89,95,59,44,126,72,124,20,94,110,45,109,100,125,5,106,134,62,56,63,8,84,74,99,49,28],"uraDoraIndicators":[84]},"publicObservation":{"playerIds":["エリカ","ヤキン","ASAPIN","ちくき"],"initScore":{"round":3,"honba":1,"tens":[12000,16000,30100,41900]},"doraIndicators":[8],"events":[{"type":"EVENT_TYPE_DRAW","who":3}]},"privateObservations":[{"initHand":{"closedTiles":[33,114,64,113,25,90,108,50,83,3,41,107,120]},"currHand":{"closedTiles":[3,25,33,41,50,64,83,90,107,108,113,114,120]}},{"who":1,"initHand":{"closedTiles":[30,4,92,55,80,96,132,26,13,86,133,37,40]},"currHand":{"closedTiles":[4,13,26,30,37,40,55,80,86,92,96,132,133]}},{"who":2,"initHand":{"closedTiles":[73,36,21,1,10,17,60,87,103,6,82,116,52]},"currHand":{"closedTiles":[1,6,10,17,21,36,52,60,73,82,87,103,116]}},{"who":3,"initHand":{"closedTiles":[2,75,111,69,128,11,115,57,51,119,47,35,123]},"drawHistory":[101],"currHand":{"closedTiles":[2,11,35,47,51,57,69,75,101,111,115,119,123,128]}}]}'
    obs = state2obs(json_str, 3)
    feature = obs.to_features("han22-v0")
    val = [
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        True,
        False,
        True,
        False,
    ]

    # index92: If t is Kyuhai and is in player 0's hand
    assert all(feature[92] == val)


def test_index_81():
    json_str = '{"who":1,"publicObservation":{"playerIds":["rule-based-0","target-player","rule-based-2","rule-based-3"],"initScore":{"round":2,"tens":[25200,24000,27300,23500]},"doraIndicators":[88],"events":[{"type":"EVENT_TYPE_DRAW","who":2},{"who":2,"tile":108},{"type":"EVENT_TYPE_DRAW","who":3},{"who":3,"tile":119},{"type":"EVENT_TYPE_DRAW"},{"tile":116},{"type":"EVENT_TYPE_DRAW","who":1},{"who":1,"tile":110},{"type":"EVENT_TYPE_DRAW","who":2},{"who":2,"tile":114},{"type":"EVENT_TYPE_DRAW","who":3},{"type":"EVENT_TYPE_TSUMOGIRI","who":3,"tile":111},{"type":"EVENT_TYPE_DRAW"},{"type":"EVENT_TYPE_TSUMOGIRI","tile":134},{"type":"EVENT_TYPE_DRAW","who":1},{"type":"EVENT_TYPE_TSUMOGIRI","who":1,"tile":122},{"type":"EVENT_TYPE_DRAW","who":2},{"who":2,"tile":39},{"type":"EVENT_TYPE_DRAW","who":3},{"who":3,"tile":84},{"type":"EVENT_TYPE_DRAW"},{"type":"EVENT_TYPE_TSUMOGIRI","tile":123},{"type":"EVENT_TYPE_DRAW","who":1},{"who":1,"tile":124},{"type":"EVENT_TYPE_DRAW","who":2},{"who":2,"tile":106},{"type":"EVENT_TYPE_DRAW","who":3},{"type":"EVENT_TYPE_TSUMOGIRI","who":3,"tile":25},{"type":"EVENT_TYPE_DRAW"},{"tile":37},{"type":"EVENT_TYPE_DRAW","who":1},{"type":"EVENT_TYPE_TSUMOGIRI","who":1,"tile":117},{"type":"EVENT_TYPE_DRAW","who":2},{"type":"EVENT_TYPE_TSUMOGIRI","who":2,"tile":36},{"type":"EVENT_TYPE_DRAW","who":3},{"type":"EVENT_TYPE_TSUMOGIRI","who":3,"tile":118},{"type":"EVENT_TYPE_DRAW"},{"tile":68},{"type":"EVENT_TYPE_DRAW","who":1},{"who":1,"tile":41},{"type":"EVENT_TYPE_DRAW","who":2},{"who":2},{"type":"EVENT_TYPE_DRAW","who":3},{"who":3,"tile":51},{"type":"EVENT_TYPE_PON","who":2,"open":19465},{"who":2,"tile":4},{"type":"EVENT_TYPE_DRAW","who":3},{"who":3,"tile":100},{"type":"EVENT_TYPE_CHI","open":60463},{"tile":42},{"type":"EVENT_TYPE_DRAW","who":1},{"who":1,"tile":69},{"type":"EVENT_TYPE_DRAW","who":2},{"type":"EVENT_TYPE_TSUMOGIRI","who":2,"tile":7},{"type":"EVENT_TYPE_PON","open":2570},{"tile":54},{"type":"EVENT_TYPE_DRAW","who":1},{"who":1,"tile":79},{"type":"EVENT_TYPE_DRAW","who":2},{"type":"EVENT_TYPE_TSUMOGIRI","who":2,"tile":120},{"type":"EVENT_TYPE_DRAW","who":3},{"type":"EVENT_TYPE_TSUMOGIRI","who":3,"tile":78},{"type":"EVENT_TYPE_DRAW"},{"type":"EVENT_TYPE_TSUMOGIRI","tile":1},{"type":"EVENT_TYPE_DRAW","who":1},{"who":1,"tile":85},{"type":"EVENT_TYPE_DRAW","who":2},{"who":2,"tile":43},{"type":"EVENT_TYPE_DRAW","who":3},{"type":"EVENT_TYPE_TSUMOGIRI","who":3,"tile":121},{"type":"EVENT_TYPE_DRAW"},{"tile":24},{"type":"EVENT_TYPE_DRAW","who":1},{"who":1,"tile":95},{"type":"EVENT_TYPE_DRAW","who":2},{"who":2,"tile":57},{"type":"EVENT_TYPE_DRAW","who":3},{"type":"EVENT_TYPE_RIICHI","who":3},{"who":3,"tile":59},{"type":"EVENT_TYPE_RIICHI_SCORE_CHANGE","who":3},{"type":"EVENT_TYPE_DRAW"},{"type":"EVENT_TYPE_TSUMOGIRI","tile":115},{"type":"EVENT_TYPE_DRAW","who":1},{"who":1,"tile":101},{"type":"EVENT_TYPE_DRAW","who":2},{"type":"EVENT_TYPE_TSUMOGIRI","who":2,"tile":70},{"type":"EVENT_TYPE_DRAW","who":3},{"type":"EVENT_TYPE_TSUMOGIRI","who":3,"tile":104},{"type":"EVENT_TYPE_DRAW"},{"type":"EVENT_TYPE_TSUMOGIRI","tile":112},{"type":"EVENT_TYPE_DRAW","who":1},{"type":"EVENT_TYPE_TSUMOGIRI","who":1,"tile":83},{"type":"EVENT_TYPE_PON","who":2,"open":31787},{"who":2,"tile":33},{"type":"EVENT_TYPE_DRAW","who":3},{"type":"EVENT_TYPE_TSUMO","who":3,"tile":67}]},"privateObservation":{"who":1,"initHand":{"closedTiles":[129,101,63,128,31,41,10,21,124,69,14,110,79]},"drawHistory":[26,122,95,117,55,85,58,13,61,44,83],"currHand":{"closedTiles":[10,13,14,21,26,31,44,55,58,61,63,128,129]}},"roundTerminal":{"finalScore":{"round":2,"tens":[22200,21000,21300,35500]},"wins":[{"who":3,"fromWho":3,"hand":{"closedTiles":[20,23,46,47,60,62,65,67,73,75,96,99,133,135]},"winTile":67,"fu":25,"ten":12000,"tenChanges":[-3000,-3000,-6000,13000],"yakus":[0,1,22,53],"fans":[1,1,2,2],"uraDoraIndicators":[16]}]},"legalActions":[{"who":1,"type":"ACTION_TYPE_DUMMY"}]}'
    obs = Observation(json_str)
    feature = obs.to_features("han22-v0")

    # index81: If at t is in player 0's hand
    # これはindex0とは異なり、tを捨てることができるかをT/Fで表現する(mjxで言うところのlegal_actionsを見ている)
    # テスト事例はゲーム終了時なので、index81にTrueは含まれないはず
    assert True not in feature[81]


def test_mjx_small_v0():
    json_str = '{"who":1,"publicObservation":{"playerIds":["rule-based-0","target-player","rule-based-2","rule-based-3"],"initScore":{"round":2,"tens":[25200,24000,27300,23500]},"doraIndicators":[88],"events":[{"type":"EVENT_TYPE_DRAW","who":2},{"who":2,"tile":108},{"type":"EVENT_TYPE_DRAW","who":3},{"who":3,"tile":119},{"type":"EVENT_TYPE_DRAW"},{"tile":116},{"type":"EVENT_TYPE_DRAW","who":1},{"who":1,"tile":110},{"type":"EVENT_TYPE_DRAW","who":2},{"who":2,"tile":114},{"type":"EVENT_TYPE_DRAW","who":3},{"type":"EVENT_TYPE_TSUMOGIRI","who":3,"tile":111},{"type":"EVENT_TYPE_DRAW"},{"type":"EVENT_TYPE_TSUMOGIRI","tile":134},{"type":"EVENT_TYPE_DRAW","who":1},{"type":"EVENT_TYPE_TSUMOGIRI","who":1,"tile":122},{"type":"EVENT_TYPE_DRAW","who":2},{"who":2,"tile":39},{"type":"EVENT_TYPE_DRAW","who":3},{"who":3,"tile":84},{"type":"EVENT_TYPE_DRAW"},{"type":"EVENT_TYPE_TSUMOGIRI","tile":123},{"type":"EVENT_TYPE_DRAW","who":1},{"who":1,"tile":124},{"type":"EVENT_TYPE_DRAW","who":2},{"who":2,"tile":106},{"type":"EVENT_TYPE_DRAW","who":3},{"type":"EVENT_TYPE_TSUMOGIRI","who":3,"tile":25},{"type":"EVENT_TYPE_DRAW"},{"tile":37},{"type":"EVENT_TYPE_DRAW","who":1},{"type":"EVENT_TYPE_TSUMOGIRI","who":1,"tile":117},{"type":"EVENT_TYPE_DRAW","who":2},{"type":"EVENT_TYPE_TSUMOGIRI","who":2,"tile":36},{"type":"EVENT_TYPE_DRAW","who":3},{"type":"EVENT_TYPE_TSUMOGIRI","who":3,"tile":118},{"type":"EVENT_TYPE_DRAW"},{"tile":68},{"type":"EVENT_TYPE_DRAW","who":1},{"who":1,"tile":41},{"type":"EVENT_TYPE_DRAW","who":2},{"who":2},{"type":"EVENT_TYPE_DRAW","who":3},{"who":3,"tile":51},{"type":"EVENT_TYPE_PON","who":2,"open":19465},{"who":2,"tile":4},{"type":"EVENT_TYPE_DRAW","who":3},{"who":3,"tile":100},{"type":"EVENT_TYPE_CHI","open":60463},{"tile":42},{"type":"EVENT_TYPE_DRAW","who":1},{"who":1,"tile":69},{"type":"EVENT_TYPE_DRAW","who":2},{"type":"EVENT_TYPE_TSUMOGIRI","who":2,"tile":7},{"type":"EVENT_TYPE_PON","open":2570},{"tile":54},{"type":"EVENT_TYPE_DRAW","who":1},{"who":1,"tile":79},{"type":"EVENT_TYPE_DRAW","who":2},{"type":"EVENT_TYPE_TSUMOGIRI","who":2,"tile":120},{"type":"EVENT_TYPE_DRAW","who":3},{"type":"EVENT_TYPE_TSUMOGIRI","who":3,"tile":78},{"type":"EVENT_TYPE_DRAW"},{"type":"EVENT_TYPE_TSUMOGIRI","tile":1},{"type":"EVENT_TYPE_DRAW","who":1},{"who":1,"tile":85},{"type":"EVENT_TYPE_DRAW","who":2},{"who":2,"tile":43},{"type":"EVENT_TYPE_DRAW","who":3},{"type":"EVENT_TYPE_TSUMOGIRI","who":3,"tile":121},{"type":"EVENT_TYPE_DRAW"},{"tile":24},{"type":"EVENT_TYPE_DRAW","who":1},{"who":1,"tile":95},{"type":"EVENT_TYPE_DRAW","who":2},{"who":2,"tile":57},{"type":"EVENT_TYPE_DRAW","who":3},{"type":"EVENT_TYPE_RIICHI","who":3},{"who":3,"tile":59},{"type":"EVENT_TYPE_RIICHI_SCORE_CHANGE","who":3},{"type":"EVENT_TYPE_DRAW"},{"type":"EVENT_TYPE_TSUMOGIRI","tile":115},{"type":"EVENT_TYPE_DRAW","who":1},{"who":1,"tile":101},{"type":"EVENT_TYPE_DRAW","who":2},{"type":"EVENT_TYPE_TSUMOGIRI","who":2,"tile":70},{"type":"EVENT_TYPE_DRAW","who":3},{"type":"EVENT_TYPE_TSUMOGIRI","who":3,"tile":104},{"type":"EVENT_TYPE_DRAW"},{"type":"EVENT_TYPE_TSUMOGIRI","tile":112},{"type":"EVENT_TYPE_DRAW","who":1},{"type":"EVENT_TYPE_TSUMOGIRI","who":1,"tile":83},{"type":"EVENT_TYPE_PON","who":2,"open":31787},{"who":2,"tile":33},{"type":"EVENT_TYPE_DRAW","who":3},{"type":"EVENT_TYPE_TSUMO","who":3,"tile":67}]},"privateObservation":{"who":1,"initHand":{"closedTiles":[129,101,63,128,31,41,10,21,124,69,14,110,79]},"drawHistory":[26,122,95,117,55,85,58,13,61,44,83],"currHand":{"closedTiles":[10,13,14,21,26,31,44,55,58,61,63,128,129]}},"roundTerminal":{"finalScore":{"round":2,"tens":[22200,21000,21300,35500]},"wins":[{"who":3,"fromWho":3,"hand":{"closedTiles":[20,23,46,47,60,62,65,67,73,75,96,99,133,135]},"winTile":67,"fu":25,"ten":12000,"tenChanges":[-3000,-3000,-6000,13000],"yakus":[0,1,22,53],"fans":[1,1,2,2],"uraDoraIndicators":[16]}]},"legalActions":[{"who":1,"type":"ACTION_TYPE_DUMMY"}]}'
    obs = Observation(json_str)

    features = obs.to_features("mjx-small-v0")
    assert features.shape == (16, 34)
    # TODO: add further tests
