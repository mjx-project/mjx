import mjx

sample_json = """{"closedTiles":[0,4,8,12,16,20,24,28,32,36,40,44,48]}"""


def test_to_json():
    hand = mjx.Hand(sample_json)
    assert hand.to_json() == sample_json


def test_shanten_number():
    hand = mjx.Hand(sample_json)
    assert hand.shanten_number() == 0

    hand = mjx.Hand("""{"closedTiles":[0,4,8,12,16,20,24,28,32,36,40,44,48,48]}""")
    assert hand.shanten_number() == -1


def test_effective_discard_types():
    hand = mjx.Hand(sample_json)
    effective_draw_types = hand.effective_draw_types()
    assert len(effective_draw_types) == 2
    assert mjx.TileType.P1 in effective_draw_types
    assert mjx.TileType.P4 in effective_draw_types


def test_closed_tiles():
    hand = mjx.Hand(sample_json)
    assert len(hand.closed_tiles()) == 13
