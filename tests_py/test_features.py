from mjx import Observation


def test_self_wind():
    json_str = '{"publicObservation":{"playerIds":["player_2","player_1","player_0","player_3"],"initScore":{"tens":[25000,25000,25000,25000]},"doraIndicators":[101],"events":[{"type":"EVENT_TYPE_DRAW"}]},"privateObservation":{"initHand":{"closedTiles":[24,3,87,124,37,42,58,134,92,82,122,18,117]},"drawHistory":[79],"currHand":{"closedTiles":[3,18,24,37,42,58,79,82,87,92,117,122,124,134]}},"legalActions":[{"tile":3},{"tile":18},{"tile":24},{"tile":37},{"tile":42},{"tile":58},{"type":"ACTION_TYPE_TSUMOGIRI","tile":79},{"tile":82},{"tile":87},{"tile":92},{"tile":117},{"tile":122},{"tile":124},{"tile":134}]}'
    obs = Observation(json_str)
    feature = obs.get_feature()

    # index78:If t is wind of self
    # 27:EW
    assert feature[78][27]
    assert not feature[78][28]


def test_table_wind():
    json_str = '{"publicObservation":{"playerIds":["player_2","player_1","player_0","player_3"],"initScore":{"tens":[25000,25000,25000,25000]},"doraIndicators":[101],"events":[{"type":"EVENT_TYPE_DRAW"}]},"privateObservation":{"initHand":{"closedTiles":[24,3,87,124,37,42,58,134,92,82,122,18,117]},"drawHistory":[79],"currHand":{"closedTiles":[3,18,24,37,42,58,79,82,87,92,117,122,124,134]}},"legalActions":[{"tile":3},{"tile":18},{"tile":24},{"tile":37},{"tile":42},{"tile":58},{"type":"ACTION_TYPE_TSUMOGIRI","tile":79},{"tile":82},{"tile":87},{"tile":92},{"tile":117},{"tile":122},{"tile":124},{"tile":134}]}'
    obs = Observation(json_str)
    feature = obs.get_feature()

    # index79: If t is wind of the table
    # 27: EW
    assert feature[79][27]
    assert not feature[79][28]


def test_dora_indicator():
    json_str = '{"publicObservation":{"playerIds":["player_2","player_1","player_0","player_3"],"initScore":{"tens":[25000,25000,25000,25000]},"doraIndicators":[101],"events":[{"type":"EVENT_TYPE_DRAW"}]},"privateObservation":{"initHand":{"closedTiles":[24,3,87,124,37,42,58,134,92,82,122,18,117]},"drawHistory":[79],"currHand":{"closedTiles":[3,18,24,37,42,58,79,82,87,92,117,122,124,134]}},"legalActions":[{"tile":3},{"tile":18},{"tile":24},{"tile":37},{"tile":42},{"tile":58},{"type":"ACTION_TYPE_TSUMOGIRI","tile":79},{"tile":82},{"tile":87},{"tile":92},{"tile":117},{"tile":122},{"tile":124},{"tile":134}]}'
    obs = Observation(json_str)
    feature = obs.get_feature()

    assert feature[70][24]


def test_dora():
    json_str = '{"publicObservation":{"playerIds":["player_2","player_1","player_0","player_3"],"initScore":{"tens":[25000,25000,25000,25000]},"doraIndicators":[101],"events":[{"type":"EVENT_TYPE_DRAW"}]},"privateObservation":{"initHand":{"closedTiles":[24,3,87,124,37,42,58,134,92,82,122,18,117]},"drawHistory":[79],"currHand":{"closedTiles":[3,18,24,37,42,58,79,82,87,92,117,122,124,134]}},"legalActions":[{"tile":3},{"tile":18},{"tile":24},{"tile":37},{"tile":42},{"tile":58},{"type":"ACTION_TYPE_TSUMOGIRI","tile":79},{"tile":82},{"tile":87},{"tile":92},{"tile":117},{"tile":122},{"tile":124},{"tile":134}]}'
    obs = Observation(json_str)
    feature = obs.get_feature()

    assert feature[74][25]


def test_closed_tiles():
    json_str = '{"publicObservation":{"playerIds":["player_2","player_1","player_0","player_3"],"initScore":{"tens":[25000,25000,25000,25000]},"doraIndicators":[101],"events":[{"type":"EVENT_TYPE_DRAW"}]},"privateObservation":{"initHand":{"closedTiles":[24,3,87,124,37,42,58,134,92,82,122,18,117]},"drawHistory":[79],"currHand":{"closedTiles":[3,18,24,37,42,58,79,82,87,92,117,122,124,134]}},"legalActions":[{"tile":3},{"tile":18},{"tile":24},{"tile":37},{"tile":42},{"tile":58},{"type":"ACTION_TYPE_TSUMOGIRI","tile":79},{"tile":82},{"tile":87},{"tile":92},{"tile":117},{"tile":122},{"tile":124},{"tile":134}]}'
    obs = Observation(json_str)
    feature = obs.get_feature()

    # index79:If t is wind of self)
    # 27:EW
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
    feature = obs.get_feature()
    assert feature[59][14]
