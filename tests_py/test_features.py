from matplotlib.pyplot import close

from mjx import Observation

json_str = '{"publicObservation":{"playerIds":["player_2","player_1","player_0","player_3"],"initScore":{"tens":[25000,25000,25000,25000]},"doraIndicators":[101],"events":[{"type":"EVENT_TYPE_DRAW"}]},"privateObservation":{"initHand":{"closedTiles":[24,3,87,124,37,42,58,134,92,82,122,18,117]},"drawHistory":[79],"currHand":{"closedTiles":[3,18,24,37,42,58,79,82,87,92,117,122,124,134]}},"legalActions":[{"tile":3},{"tile":18},{"tile":24},{"tile":37},{"tile":42},{"tile":58},{"type":"ACTION_TYPE_TSUMOGIRI","tile":79},{"tile":82},{"tile":87},{"tile":92},{"tile":117},{"tile":122},{"tile":124},{"tile":134}]}'


def test_self_wind():

    obs = Observation(json_str)
    feature = obs.get_feature()

    # index78:If t is wind of self
    # 27:EW
    assert feature[78][27]
    assert not feature[78][28]


def test_table_wind():

    obs = Observation(json_str)
    feature = obs.get_feature()

    # index79: If t is wind of the table
    # 27: EW
    assert feature[79][27]
    assert not feature[79][28]


def test_closed_tiles():

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
