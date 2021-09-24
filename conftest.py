def pytest_ignore_collect(path):
    if "pb2" in str(path):
        return True
    if "main.py" in str(path):
        return True
    if "pybind11" in str(path):
        return True
    if "external" in str(path):
        return True
    return False
