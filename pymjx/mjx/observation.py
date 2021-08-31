class Observation:
    def __init__(self, cpp_obj):
        self._cpp_obj = cpp_obj

    def to_feature(self, version):
        return self._cpp_obj.to_feature(version)

    def legal_actions(self):
        return self._cpp_obj.legal_actions()

    def action_mask(self):
        return self._cpp_obj.action_mask()