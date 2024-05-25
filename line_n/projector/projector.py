from typing import List

import numpy as np
import matplotlib.pyplot as plt

from line_n import Canvas
from line_n.projector.utils import connect_points


class Projector:
    def __init__(
        self,
        canvas: Canvas,
        image: np.ndarray,
        max_optimizer_steps,
        transform=None,
        transform_kwargs=None,
    ):
        self.canvas = canvas
        self.invert_brightness = self.canvas.dark_background

        self.image = self._setup_image(image)

        self.transformer = self._setup_transformer(transform, transform_kwargs)
        self.optimizer = self._setup_optimizer(max_optimizer_steps)

    def _setup_image(self, image: np.ndarray) -> np.ndarray:
        if image.dtype == np.uint8:
            print("Casting image to float, setting max value to 1.0")
            image = image.astype(float) / 255
        elif image.dtype != float or image.max() > 1:
            raise TypeError(
                "Image must be type uint8 or float lower than or equal to 1.0"
            )
        return 1 - image if self.invert_brightness else image

    def _setup_transformer(self, transform, kwargs):
        kwargs = kwargs or {}

        def identity(x):
            return x

        def transformer(x):
            return transform(x, **kwargs)

        return identity if transform is None else transformer

    def _setup_optimizer(self, max_steps):
        return {
            "stop": False,
            "current_metric": -np.inf,
            "steps": 0,
            "max_steps": max_steps,
            "metric_history": [],
            "has_been_optimized": False,
        }

    @property
    def best_edges(self):
        return (
            self.optimize_edges()
            if not self.optimizer["has_been_optimized"]
            else self._best_edges
        )

    def get_spectrum(self, image):
        dot = np.empty(len(self.canvas.base.keys()), dtype=float)
        for edge_idx in range(dot.size):
            edge_img = self.canvas.edges2image(edge_idx)
            dot[edge_idx] = np.sum(image * edge_img)
        return dot

    def _project(self, image):
        trans_input_img = self.transformer(image)
        trans_self_img = self.transformer(self.image)
        return np.sum(trans_input_img * trans_self_img)

    def optimize_edges(self):
        new_img = self.image.copy()
        spectrum = self.get_spectrum(self.image)
        self._best_edges = []
        max_dot = spectrum.argmax()

        while not self.optimizer["stop"]:
            self._best_edges.append(max_dot)
            yield max_dot
            new_img -= self.canvas.edges2image(max_dot)

            spectrum = self.get_spectrum(new_img)
            max_dot = spectrum.argmax()
            self.optimizer_step()

        return None

    def optimizer_step(self):
        self.optimizer["has_been_optimized"] = True
        image_so_far = self.canvas.edges2image(*self._best_edges)
        metric = self._project(image_so_far)
        self.optimizer["metric_history"].append(metric)
        metric_diff = metric - self.optimizer["current_metric"]
        self.optimizer["current_metric"] = metric
        if metric_diff <= 0:
            self.optimizer["stop"] = True

        elif self.optimizer["steps"] >= self.optimizer["max_steps"]:
            print("Max steps reached")
            self.optimizer["stop"] = True

        self.optimizer["steps"] += 1

    def plot(self, ax=None, **kwargs):
        edges = self.best_edges
        self.canvas.plot(*edges, ax=ax, **kwargs)
