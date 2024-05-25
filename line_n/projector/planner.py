from typing import List, Tuple

import numpy as np

from line_n.canvas.canvas import Canvas
from line_n.projector.utils import connect_points


class Planner:
    def __init__(self, canvas: Canvas, thread_projection: List[int]):
        self.canvas = canvas
        self.thread_projection = thread_projection
        self._plan = self._get_plan()

    def _get_plan(self):
        adj_matrix = self._get_frame_adjacency()
        return connect_points(adj_matrix=adj_matrix)

    def _get_frame_adjacency(self) -> np.ndarray:
        adj_matrix = np.zeros(
            (len(self.canvas.frame.points), len(self.canvas.frame.points))
        )
        for edge_idx, (point_1, point_2) in self.canvas.edge_map:
            if edge_idx in self.thread_projection:
                idx_p1 = np.nonzero(
                    np.equal(self.canvas.frame_px, point_1).all(axis=1)
                )[0]
                idx_p2 = np.nonzero(
                    np.equal(self.canvas.frame_px, point_2).all(axis=1)
                )[0]
                adj_matrix[idx_p1, idx_p2] += 1
                adj_matrix[idx_p2, idx_p1] += 1

        return adj_matrix

    @property
    def plan(self):
        return self._plan

    @property
    def total_thread(self):
        if not self.optimizer["has_been_optimized"]:
            raise NotOptimizedError(
                "Projector needs to be optimized before retrieving the total used thread."
            )
        total_thread = 0
        for point_1, point_2 in self.plan:
            dist = self.canvas.frame.points[point_1] - self.canvas.frame.points[point_2]
            total_thread += np.linalg.norm(dist)
        return total_thread

    def print(self):
        thread_color = f"Thread color: {self.canvas.thread.color_name}"

        def edge_direction(pair: Tuple[int, int]):
            return f"{pair[0]}\t->\t{pair[1]}"

        header = "\tStart\t->\tEnd\n____________________________"
        plan_txt = []
        for step, point_pair in enumerate(self.plan):
            plan_txt.append(f"{step}:\t{edge_direction(point_pair)}")

        recipe = "\n".join([thread_color, header, *plan_txt])
        return recipe

    def save(self, filename: str):
        recipe = self.print()
        with open(filename, "w") as file:
            file.write(recipe)
