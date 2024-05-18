# Canvas exceptions
class ColorIncompatibilityError(Exception):
    def __init__(self, message: str):
        self.message = message


# Thread exceptions
class ColorMapError(Exception):
    def __init__(self, message: str):
        self.message = message
