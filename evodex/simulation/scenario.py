class Scenario:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def __repr__(self):
        return f"Scenario(name={self.name}, description={self.description})"