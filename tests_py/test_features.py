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
    json_str = '{"publicObservation":{"playerIds":["player_2","player_1","player_0","player_3"],"initScore":{"tens":[25000,25000,25000,25000]},"doraIndicators":[101],"events":[{"type":"EVENT_TYPE_DRAW"}]},"privateObservation":{"initHand":{"closedTiles":[24,3,87,124,37,42,58,134,92,82,122,18,117]},"drawHistory":[79],"currHand":{"closedTiles":[3,18,24,37,42,58,79,82,87,92,117,122,124,134]}},"legalActions":[{"tile":3},{"tile":18},{"tile":24},{"tile":37},{"tile":42},{"tile":58},{"type":"ACTION_TYPE_TSUMOGIRI","tile":79},{"tile":82},{"tile":87},{"tile":92},{"tile":117},{"tile":122},{"tile":124},{"tile":134}]}'
    obs = Observation(json_str)
    feature = obs.to_features("han22-v0")

    # index70: If t is Dora indicator
    # 24: 7s
    assert feature[70][24]


def test_dora():
    json_str = '{"publicObservation":{"playerIds":["player_2","player_1","player_0","player_3"],"initScore":{"tens":[25000,25000,25000,25000]},"doraIndicators":[101],"events":[{"type":"EVENT_TYPE_DRAW"}]},"privateObservation":{"initHand":{"closedTiles":[24,3,87,124,37,42,58,134,92,82,122,18,117]},"drawHistory":[79],"currHand":{"closedTiles":[3,18,24,37,42,58,79,82,87,92,117,122,124,134]}},"legalActions":[{"tile":3},{"tile":18},{"tile":24},{"tile":37},{"tile":42},{"tile":58},{"type":"ACTION_TYPE_TSUMOGIRI","tile":79},{"tile":82},{"tile":87},{"tile":92},{"tile":117},{"tile":122},{"tile":124},{"tile":134}]}'
    obs = Observation(json_str)
    feature = obs.to_features("han22-v0")

    # index74: If t is Dora
    # 24: 8s
    assert feature[74][25]


def test_closed_tiles():
    json_str = '{"publicObservation":{"playerIds":["player_2","player_1","player_0","player_3"],"initScore":{"tens":[25000,25000,25000,25000]},"doraIndicators":[101],"events":[{"type":"EVENT_TYPE_DRAW"}]},"privateObservation":{"initHand":{"closedTiles":[24,3,87,124,37,42,58,134,92,82,122,18,117]},"drawHistory":[79],"currHand":{"closedTiles":[3,18,24,37,42,58,79,82,87,92,117,122,124,134]}},"legalActions":[{"tile":3},{"tile":18},{"tile":24},{"tile":37},{"tile":42},{"tile":58},{"type":"ACTION_TYPE_TSUMOGIRI","tile":79},{"tile":82},{"tile":87},{"tile":92},{"tile":117},{"tile":122},{"tile":124},{"tile":134}]}'
    obs = Observation(json_str)
    feature = obs.to_features("han22-v0")

    # index0: If player 0 has >= 1 t in hand
    closed = feature[0]
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
    assert all(closed == val)


def test_riichi():
    json_str = '{"who":1,"publicObservation":{"playerIds":["rule-based-0","target-player","rule-based-2","rule-based-3"],"initScore":{"round":2,"tens":[25200,24000,27300,23500]},"doraIndicators":[88],"events":[{"type":"EVENT_TYPE_DRAW","who":2},{"who":2,"tile":108},{"type":"EVENT_TYPE_DRAW","who":3},{"who":3,"tile":119},{"type":"EVENT_TYPE_DRAW"},{"tile":116},{"type":"EVENT_TYPE_DRAW","who":1},{"who":1,"tile":110},{"type":"EVENT_TYPE_DRAW","who":2},{"who":2,"tile":114},{"type":"EVENT_TYPE_DRAW","who":3},{"type":"EVENT_TYPE_TSUMOGIRI","who":3,"tile":111},{"type":"EVENT_TYPE_DRAW"},{"type":"EVENT_TYPE_TSUMOGIRI","tile":134},{"type":"EVENT_TYPE_DRAW","who":1},{"type":"EVENT_TYPE_TSUMOGIRI","who":1,"tile":122},{"type":"EVENT_TYPE_DRAW","who":2},{"who":2,"tile":39},{"type":"EVENT_TYPE_DRAW","who":3},{"who":3,"tile":84},{"type":"EVENT_TYPE_DRAW"},{"type":"EVENT_TYPE_TSUMOGIRI","tile":123},{"type":"EVENT_TYPE_DRAW","who":1},{"who":1,"tile":124},{"type":"EVENT_TYPE_DRAW","who":2},{"who":2,"tile":106},{"type":"EVENT_TYPE_DRAW","who":3},{"type":"EVENT_TYPE_TSUMOGIRI","who":3,"tile":25},{"type":"EVENT_TYPE_DRAW"},{"tile":37},{"type":"EVENT_TYPE_DRAW","who":1},{"type":"EVENT_TYPE_TSUMOGIRI","who":1,"tile":117},{"type":"EVENT_TYPE_DRAW","who":2},{"type":"EVENT_TYPE_TSUMOGIRI","who":2,"tile":36},{"type":"EVENT_TYPE_DRAW","who":3},{"type":"EVENT_TYPE_TSUMOGIRI","who":3,"tile":118},{"type":"EVENT_TYPE_DRAW"},{"tile":68},{"type":"EVENT_TYPE_DRAW","who":1},{"who":1,"tile":41},{"type":"EVENT_TYPE_DRAW","who":2},{"who":2},{"type":"EVENT_TYPE_DRAW","who":3},{"who":3,"tile":51},{"type":"EVENT_TYPE_PON","who":2,"open":19465},{"who":2,"tile":4},{"type":"EVENT_TYPE_DRAW","who":3},{"who":3,"tile":100},{"type":"EVENT_TYPE_CHI","open":60463},{"tile":42},{"type":"EVENT_TYPE_DRAW","who":1},{"who":1,"tile":69},{"type":"EVENT_TYPE_DRAW","who":2},{"type":"EVENT_TYPE_TSUMOGIRI","who":2,"tile":7},{"type":"EVENT_TYPE_PON","open":2570},{"tile":54},{"type":"EVENT_TYPE_DRAW","who":1},{"who":1,"tile":79},{"type":"EVENT_TYPE_DRAW","who":2},{"type":"EVENT_TYPE_TSUMOGIRI","who":2,"tile":120},{"type":"EVENT_TYPE_DRAW","who":3},{"type":"EVENT_TYPE_TSUMOGIRI","who":3,"tile":78},{"type":"EVENT_TYPE_DRAW"},{"type":"EVENT_TYPE_TSUMOGIRI","tile":1},{"type":"EVENT_TYPE_DRAW","who":1},{"who":1,"tile":85},{"type":"EVENT_TYPE_DRAW","who":2},{"who":2,"tile":43},{"type":"EVENT_TYPE_DRAW","who":3},{"type":"EVENT_TYPE_TSUMOGIRI","who":3,"tile":121},{"type":"EVENT_TYPE_DRAW"},{"tile":24},{"type":"EVENT_TYPE_DRAW","who":1},{"who":1,"tile":95},{"type":"EVENT_TYPE_DRAW","who":2},{"who":2,"tile":57},{"type":"EVENT_TYPE_DRAW","who":3},{"type":"EVENT_TYPE_RIICHI","who":3},{"who":3,"tile":59},{"type":"EVENT_TYPE_RIICHI_SCORE_CHANGE","who":3},{"type":"EVENT_TYPE_DRAW"},{"type":"EVENT_TYPE_TSUMOGIRI","tile":115},{"type":"EVENT_TYPE_DRAW","who":1},{"who":1,"tile":101},{"type":"EVENT_TYPE_DRAW","who":2},{"type":"EVENT_TYPE_TSUMOGIRI","who":2,"tile":70},{"type":"EVENT_TYPE_DRAW","who":3},{"type":"EVENT_TYPE_TSUMOGIRI","who":3,"tile":104},{"type":"EVENT_TYPE_DRAW"},{"type":"EVENT_TYPE_TSUMOGIRI","tile":112},{"type":"EVENT_TYPE_DRAW","who":1},{"type":"EVENT_TYPE_TSUMOGIRI","who":1,"tile":83},{"type":"EVENT_TYPE_PON","who":2,"open":31787},{"who":2,"tile":33},{"type":"EVENT_TYPE_DRAW","who":3},{"type":"EVENT_TYPE_TSUMO","who":3,"tile":67}]},"privateObservation":{"who":1,"initHand":{"closedTiles":[129,101,63,128,31,41,10,21,124,69,14,110,79]},"drawHistory":[26,122,95,117,55,85,58,13,61,44,83],"currHand":{"closedTiles":[10,13,14,21,26,31,44,55,58,61,63,128,129]}},"roundTerminal":{"finalScore":{"round":2,"tens":[22200,21000,21300,35500]},"wins":[{"who":3,"fromWho":3,"hand":{"closedTiles":[20,23,46,47,60,62,65,67,73,75,96,99,133,135]},"winTile":67,"fu":25,"ten":12000,"tenChanges":[-3000,-3000,-6000,13000],"yakus":[0,1,22,53],"fans":[1,1,2,2],"uraDoraIndicators":[16]}]},"legalActions":[{"who":1,"type":"ACTION_TYPE_DUMMY"}]}'
    obs = Observation(json_str)
    feature = obs.to_features("han22-v0")

    # index59: If player 1 has discarded t to announce Riichi
    # 14: 2p
    assert feature[59][14]


def test_three_tiles_in_hand():
    json_str = '{"who":1,"publicObservation":{"playerIds":["player_1","player_2","player_0","player_3"],"initScore":{"tens":[25000,25000,25000,25000]},"doraIndicators":[6],"events":[{"type":"EVENT_TYPE_DRAW"},{"tile":27},{"type":"EVENT_TYPE_CHI","who":1,"open":16631},{"who":1,"tile":41},{"type":"EVENT_TYPE_DRAW","who":2},{"who":2,"tile":131},{"type":"EVENT_TYPE_DRAW","who":3},{"who":3,"tile":120},{"type":"EVENT_TYPE_DRAW"},{"tile":35},{"type":"EVENT_TYPE_DRAW","who":1},{"who":1,"tile":107},{"type":"EVENT_TYPE_DRAW","who":2},{"who":2,"tile":74},{"type":"EVENT_TYPE_DRAW","who":3},{"who":3,"tile":46},{"type":"EVENT_TYPE_DRAW"},{"tile":127},{"type":"EVENT_TYPE_DRAW","who":1},{"who":1,"tile":4},{"type":"EVENT_TYPE_DRAW","who":2},{"who":2,"tile":28},{"type":"EVENT_TYPE_DRAW","who":3},{"type":"EVENT_TYPE_TSUMOGIRI","who":3,"tile":1},{"type":"EVENT_TYPE_DRAW"},{"tile":75},{"type":"EVENT_TYPE_CHI","who":1,"open":43263},{"who":1,"tile":16},{"type":"EVENT_TYPE_DRAW","who":2},{"who":2,"tile":63},{"type":"EVENT_TYPE_DRAW","who":3},{"who":3,"tile":2},{"type":"EVENT_TYPE_DRAW"},{"tile":94},{"type":"EVENT_TYPE_DRAW","who":1},{"who":1,"tile":82},{"type":"EVENT_TYPE_DRAW","who":2},{"who":2,"tile":117},{"type":"EVENT_TYPE_DRAW","who":3},{"who":3,"tile":62},{"type":"EVENT_TYPE_DRAW"},{"tile":31},{"type":"EVENT_TYPE_DRAW","who":1},{"who":1,"tile":129},{"type":"EVENT_TYPE_DRAW","who":2},{"who":2,"tile":135},{"type":"EVENT_TYPE_DRAW","who":3},{"who":3,"tile":36},{"type":"EVENT_TYPE_DRAW"},{"tile":80},{"type":"EVENT_TYPE_DRAW","who":1},{"who":1,"tile":14},{"type":"EVENT_TYPE_DRAW","who":2},{"who":2,"tile":126},{"type":"EVENT_TYPE_DRAW","who":3},{"who":3,"tile":21},{"type":"EVENT_TYPE_DRAW"},{"tile":57},{"type":"EVENT_TYPE_DRAW","who":1},{"who":1,"tile":64},{"type":"EVENT_TYPE_DRAW","who":2},{"type":"EVENT_TYPE_TSUMOGIRI","who":2,"tile":130},{"type":"EVENT_TYPE_DRAW","who":3},{"who":3,"tile":47},{"type":"EVENT_TYPE_DRAW"},{"tile":110},{"type":"EVENT_TYPE_DRAW","who":1},{"who":1,"tile":66},{"type":"EVENT_TYPE_DRAW","who":2},{"type":"EVENT_TYPE_TSUMOGIRI","who":2,"tile":70},{"type":"EVENT_TYPE_DRAW","who":3},{"who":3,"tile":92},{"type":"EVENT_TYPE_DRAW"},{"tile":105},{"type":"EVENT_TYPE_DRAW","who":1}]},"privateObservation":{"who":1,"initHand":{"closedTiles":[81,107,79,82,41,99,4,14,22,97,29,66,129]},"drawHistory":[16,39,64,122,30,98,119,15],"currHand":{"closedTiles":[15,30,39,97,98,99,119,122],"opens":[16631,43263]}},"legalActions":[{"type":"ACTION_TYPE_TSUMOGIRI","who":1,"tile":15},{"who":1,"tile":30},{"who":1,"tile":39},{"who":1,"tile":97},{"who":1,"tile":119},{"who":1,"tile":122}]}'
    obs = Observation(json_str)
    feature = obs.to_features("han22-v0")

    # index2: If player 0 has >= 2 t in hand
    # 24: 2p
    assert feature[2][24]


def test_kyuuhai():
    json_str = '{"hiddenState":{"wall":[2,75,111,69,33,114,64,113,30,4,92,55,73,36,21,1,128,11,115,57,25,90,108,50,80,96,132,26,10,17,60,87,51,119,47,35,83,3,41,107,13,86,133,37,103,6,82,116,123,120,40,52,101,18,61,23,77,67,27,131,135,31,0,112,29,121,93,78,104,19,38,79,24,12,105,68,118,65,66,22,70,34,91,122,97,46,130,39,71,43,129,127,32,9,14,16,98,88,48,58,85,54,81,15,102,7,42,53,76,117,89,95,59,44,126,72,124,20,94,110,45,109,100,125,5,106,134,62,56,63,8,84,74,99,49,28],"uraDoraIndicators":[84]},"publicObservation":{"playerIds":["エリカ","ヤキン","ASAPIN","ちくき"],"initScore":{"round":3,"honba":1,"tens":[12000,16000,30100,41900]},"doraIndicators":[8],"events":[{"type":"EVENT_TYPE_DRAW","who":3}]},"privateObservations":[{"initHand":{"closedTiles":[33,114,64,113,25,90,108,50,83,3,41,107,120]},"currHand":{"closedTiles":[3,25,33,41,50,64,83,90,107,108,113,114,120]}},{"who":1,"initHand":{"closedTiles":[30,4,92,55,80,96,132,26,13,86,133,37,40]},"currHand":{"closedTiles":[4,13,26,30,37,40,55,80,86,92,96,132,133]}},{"who":2,"initHand":{"closedTiles":[73,36,21,1,10,17,60,87,103,6,82,116,52]},"currHand":{"closedTiles":[1,6,10,17,21,36,52,60,73,82,87,103,116]}},{"who":3,"initHand":{"closedTiles":[2,75,111,69,128,11,115,57,51,119,47,35,123]},"drawHistory":[101],"currHand":{"closedTiles":[2,11,35,47,51,57,69,75,101,111,115,119,123,128]}}]}'
    obs = state2obs(json_str, 3)
    feature = obs.to_features("han22-v0")
    val = [
        True,
        False,
        True,
        False,
        False,
        False,
        False,
        False,
        True,
        False,
        False,
        True,
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
        False,
        False,
        False,
        True,
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
