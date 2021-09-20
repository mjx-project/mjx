import mjx


def test_Action():
    json_str = '{"gameId":"b2a9fa48-44f1-460b-8b0a-e75d336c2bb1","tile":3}'
    action = mjx.Action(json_str)
    restored = action.to_json()
    assert json_str == restored

def test_to_json():
    json_str = '{"gameId":"b2a9fa48-44f1-460b-8b0a-e75d336c2bb1","tile":3}'
    action = mjx.Action(json_str)
    restored = action.to_json()
    assert json_str == restored

def test_to_idx():
    json_str = '{"gameId":"b2a9fa48-44f1-460b-8b0a-e75d336c2bb1","tile":10}'
    action = mjx.Action(json_str)
    idx = action.to_idx()
    assert idx == 2

if __name__ == '__main__':
    test_Action()
    test_to_json()
    test_to_idx()