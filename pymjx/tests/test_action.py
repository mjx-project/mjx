import mjx


def test_Action():
    json_str = ""
    action = mjx.Action(json_str)
    restored = action.to_json()
    assert json_str == restored