from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import numpy as np


class _Frame(ABC):

    def __init__(self, points: np.ndarray):
        """ """
        self._validate_points(points)
        self._points = np.unique(points, axis=0)

    @classmethod
    @abstractmethod
    def generate_points(cls, *args, **kwargs) -> np.ndarray:
        """"""

    def _validate_points(self, points: np.ndarray):
        if not isinstance(points, np.ndarray):
            raise TypeError(f"Points must be saved as a numpy array")

        num_points, num_coords = points.shape
        if num_coords != 2:
            raise ValueError(
                f"Points array must be of shape [num_points, 2], got {points.shape} instead"
            )

    @property
    def points(self) -> np.ndarray:
        return self._points


class SquareFrame(_Frame):
    def __init__(self, width, height, mark_density):
        step = 1 / mark_density
        x_vec = np.arange(0, width + step, step)
        y_vec = np.arange(0, height + step, step)
        top_edge = np.stack([x_vec, np.zeros(x_vec.size)], axis=1)
        right_edge = np.stack([width * np.ones(y_vec.size - 1), y_vec[1:]], axis=1)
        bottom_edge = np.stack(
            [x_vec[-2::-1], height * np.ones(x_vec.size - 1)], axis=1
        )
        left_edge = np.stack([np.zeros(y_vec.size - 2), y_vec[-2:0:-1]], axis=1)
        points = self.generate_points(
            top_edge=top_edge,
            right_edge=right_edge,
            bottom_edge=bottom_edge,
            left_edge=left_edge,
        )
        super().__init__(points=points)

    @classmethod
    def generate_points(
        cls,
        top_edge: np.ndarray,
        right_edge: np.ndarray,
        bottom_edge: np.ndarray,
        left_edge: np.ndarray,
    ) -> np.ndarray:
        return np.concatenate([top_edge, right_edge, bottom_edge, left_edge], axis=0)


class Frame(_Frame):
    def __init__(self, points: List[Union[_Frame, Tuple[float, float]]]):
        if not isinstance(points, list):
            raise TypeError(
                f"Points must be a list, got {points.__class__.__name__} instead."
            )
        for i, item in enumerate(points):
            if not isinstance(item, (list, tuple)) and not issubclass(
                item.__class__, _Frame
            ):
                raise TypeError(
                    f"All `points` must be tuple of integers or subclass of _Frame, item {i} is {item.__class__.__name__} instead"
                )

        points_list = self.generate_points(points)
        super().__init__(points=points_list)

    @classmethod
    def generate_points(
        cls, points: List[Union[_Frame, Tuple[float, float]]]
    ) -> np.ndarray:
        pts_idxs = [
            i for i, item in enumerate(points) if isinstance(item, (list, tuple))
        ]
        frame_idxs = list(set(range(len(points))).difference(set(pts_idxs)))
        pts_list = np.array([points[i] for i in pts_idxs])
        frames_pts_list = np.concatenate([points[i].points for i in frame_idxs], axis=0)
        return np.concatenate([pts_list, frames_pts_list], axis=0)
