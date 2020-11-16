def pytest_ignore_collect(path):
    if "mjproto" in str(path):
        return True
    if "main.py" in str(path):
        return True
