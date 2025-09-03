


class Retreiever:
    def __init__(self,documentpath):
        self.document = []

    def add_rule(self, rule):
        self.rules.append(rule)

    def apply_rules(self, data):
        for rule in self.rules:
            data = rule(data)
        return data