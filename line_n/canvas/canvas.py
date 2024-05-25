from itertools import combinations
from typing import Tuple, Iterable, Dict

import numpy as np
import matplotlib.pyplot as plt

from line_n.canvas.frame import _Frame
from line_n.canvas.thread import _Thread
from line_n.canvas.utils import bresenham, get_color_brightness
from line_n.exceptions.canvas_exceptions import ColorIncompatibilityError


class Canvas:
    def __init__(
        self,
        frame: _Frame,
        thread: _Thread,
        si_units=1e-2,
    ):
        self.frame = frame
        self.thread = thread
        self.dark_background = get_color_brightness(self.thread.color) > 0.5

        self.frame_px = (frame.points / (thread.width)).astype(int)
        self.base = self._get_base_edges()
        self.background_color = self._set_colors()

    @property
    def size(self) -> Tuple[int, int]:
        return self.frame_px.max(axis=0).astype(int) + 1

    @property
    def edge_map(self) -> Iterable[Tuple[int, Tuple]]:
        return (
            (edge_idx, point_pair)
            for edge_idx, point_pair in enumerate(combinations(self.frame_px, 2))
        )

    def _get_base_edges(self) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        # [K, H, W]
        base = {}
        for k, edge in self.edge_map:
            point1, point2 = edge
            (x, y) = bresenham(*point1, *point2)
            base[k] = y, x

        return base

    def _set_colors(self) -> int:
        if (self.dark_background and self.thread.color_name == "dark") or (
            not self.dark_background and self.thread.color_name == "bright"
        ):
            raise ColorIncompatibilityError(
                "Canvas background can't be the same color as the thread"
            )
        return int(not self.dark_background)

    def plot(
        self,
        *edges: int,
        show_frame: bool = True,
        enumerate_frame: bool = True,
        ax: plt.Axes = None,
        **kwargs
    ) -> None:
        cmap = "binary_r" if self.dark_background else "binary"
        if ax:
            plt.sca(ax)
        xs = self.frame_px[:, 0]
        ys = self.frame_px[:, 1]

        if show_frame:
            plt.scatter(xs, ys, **kwargs)
            if enumerate_frame:
                [plt.text(xs[i], ys[i], str(i)) for i in range(len(ys))]

        all_edges = self.edges2image(*edges)
        plt.imshow(all_edges, cmap=cmap, vmin=0, vmax=1)
        plt.xticks(xs, xs * self.thread.width)
        plt.yticks(ys, ys * self.thread.width)
        plt.gca().set_aspect("equal")

    def edges2image(self, *edges: int) -> np.ndarray:
        def edge2image(edge_idx: int) -> np.ndarray:
            y_vec, x_vec = self.base[edge_idx]
            image = np.zeros(self.size[[1, 0]], dtype=float)
            image[y_vec, x_vec] = self.thread.alpha
            return image

        if len(edges) == 0:
            return np.zeros(self.size[[1, 0]])

        return np.sum([edge2image(edge) for edge in edges], axis=0).clip(min=0, max=1)
