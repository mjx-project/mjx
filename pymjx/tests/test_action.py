import mjx


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

def test_to_idx():
    json_str = '{"gameId":"xxx","tile":10}'
    action = mjx.Action(json_str)
    idx = action.to_idx()
    assert idx == 2

def test_select_from():
    legal_actions = [mjx.Action('{"gameId":"xxx","tile":0}'), mjx.Action('{"gameId":"xxx","tile":10}')]
    action = mjx.Action.select_from(0, legal_actions)
    action.to_json() == '{"gameId":"xxx","tile":0}'


if __name__ == '__main__':
    test_Action()
    test_to_json()
    test_to_idx()
    test_select_from()