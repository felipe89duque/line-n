import numpy as np

from line_n import Canvas


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

        self.image = self._setup_image(image)
        self.transformer = self._setup_transformer(transform, transform_kwargs)
        self.optimizer = self._setup_optimizer(max_optimizer_steps)

    def _setup_image(self, image: np.ndarray) -> np.ndarray:
        if image.dtype == np.uint8:
            print("Casting image to float, setting max value to 1.0")
            return image.astype(float) / 255
        elif image.dtype != float or image.max() > 1:
            raise TypeError(
                "Image must be type uint8 or float lower than or equal to 1.0"
            )
        return image

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
        }

    @property
    def pruned_base(self):
        return [edge for edge in self.prune_edges()]

    def get_spectrum(self, image):
        dot = np.empty(len(self.canvas.base.keys()), dtype=float)
        for edge_idx in range(dot.size):
            edge_img = self.canvas.edge2base(edge_idx)
            dot[edge_idx] = np.sum(image * edge_img)
        return dot

    def project(self, image):
        trans_input_img = transformer(image)
        trans_self_img = transformer(self.image)
        return np.sum(trans_input_img * trans_self_img)

    def prune_edges(self):
        new_img = self.image.copy()
        spectrum = self.get_spectrum(self.image)
        self.best_edges = []
        max_dot = spectrum.argmax()

        # while (new_img.clip(min=0).sum() > self.image.sum()*self.energy_threshold) and (spectrum[max_dot] >0):
        while not self.stop:
            self.best_edges.append(max_dot)
            yield max_dot
            new_img -= self.canvas.edges2image(max_dot) * self.canvas.thread.alpha

            spectrum = self.get_spectrum(new_img)
            max_dot = spectrum.argmax()
            self.optimiser_step()

            # print(new_img.clip(min=0).sum(),self.image.sum(),self.image.sum()*self.energy_threshold,spectrum[max_dot])

        return None

    def optimiser_step(self):
        image = self.canvas.edges2image(*self.best_edges)
        metric = self.project(image)
        self.optimizer["metric_history"].append(metric)
        metric_diff = metric - self.optimizer["current_metric"]
        self.optimizer["current_metric"] = metric
        if metric_diff <= 0:
            self.optimizer["stop"] = True

        elif self.optimizer["steps"] >= self.optimizer["max_steps"]:
            print("Max steps reached")
            self.optimizer["stop"] = True

        self.optimizer["steps"] += 1

    def draw(self, ax=None, **kwargs):
        ax = ax or plt.axes()

        edges = self.prune_edges()
        projection = np.zeros(self.canvas.size[[1, 0]])
        for edge in edges:
            projection += self.canvas.edges2image(edge) * self.canvas.thread.alpha
        projection = (projection.clip(max=1) * 255 // projection.max()).astype(np.uint8)
        ax.imshow(projection, **kwargs)
