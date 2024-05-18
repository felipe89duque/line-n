from line_n.exceptions.canvas_exceptions import ColorMapError


class _Thread:
    ACCEPTABLE_COLORS = []
    COLOR_MAP = {}

    def __init__(self, width: float, alpha: float, color: str, si_units: float = 1e-2):
        self._alpha = alpha
        self._width = width
        self._validate_class()
        if color not in self.ACCEPTABLE_COLORS:
            raise ValueError(
                f"Color {color} not in ACCEPTABLE_COLORS: {self.ACCEPTABLE_COLORS}"
            )
        self._color_name = color
        self._color = self.COLOR_MAP[color]

    @property
    def alpha(self):
        return self._alpha

    @property
    def width(self):
        return self._width

    @property
    def color(self):
        return self._color

    @property
    def color_name(self):
        return self._color_name

    def _validate_class(self):
        if len(set(self.COLOR_MAP.keys()) ^ set(self.ACCEPTABLE_COLORS)) > 0:
            raise ColorMapError("ACCEPTABLE_COLORS and COLOR_MAP don't share all keys")


class BinaryThread(_Thread):
    ACCEPTABLE_COLORS = ["dark", "bright"]
    COLOR_MAP = {"dark": 0, "bright": 1}
