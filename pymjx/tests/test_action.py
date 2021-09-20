import mjx


def test_Action():
    json_str = '{"gameId":"b2a9fa48-44f1-460b-8b0a-e75d336c2bb1","tile":3}'
    action = mjx.Action(json_str)
    restored = action._cpp_obj.to_json()
    assert json_str == restored


if __name__ == '__main__':
    test_Action()