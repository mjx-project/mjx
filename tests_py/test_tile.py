import mjx


def test_id():
    tile = mjx.Tile(0)
    assert tile.id() == 0
    tile = mjx.Tile(135)
    assert tile.id() == 135


def test_type():
    tile: mjx.Tile = mjx.Tile(0)
    assert tile.type() == mjx.TileType.M1
    tile = mjx.Tile(135)
    assert tile.type() == mjx.TileType.RD


def test_num():
    tile = mjx.Tile(0)
    assert tile.num() == 1
    tile = mjx.Tile(135)
    assert tile.num() is None


def test_is_red():
    tile = mjx.Tile(16)
    assert tile.is_red()
    assert tile.num() == 5
    tile = mjx.Tile(52)
    assert tile.is_red()
    assert tile.num() == 5
    tile = mjx.Tile(88)
    assert tile.is_red()
    assert tile.num() == 5

    tile = mjx.Tile(0)
    assert not tile.is_red()
    tile = mjx.Tile(135)
    assert not tile.is_red()
