import mjx

sample_json = """{"closedTiles":[0,1,2,3,4,5,6,7,8,9,10,11,12]}"""


def test_to_json():
    hand = mjx.Hand(sample_json)
    assert hand.to_json() == sample_json


def test_shanten_number():
    hand = mjx.Hand(sample_json)
    assert hand.shanten_number() == 0

    hand = mjx.Hand("""{"closedTiles":[0,1,2,3,4,5,6,7,8,9,10,11,12,12]}""")
    assert hand.shanten_number() == -1


def test_closed_tiles():
    hand = mjx.Hand(sample_json)
    assert len(hand.closed_tiles()) == 13
