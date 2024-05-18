import numpy as np


def bresenham(x0, y0, x1, y1):
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    error = dx + dy

    x_vec = []
    y_vec = []
    while True:
        x_vec.append(x0)
        y_vec.append(y0)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * error
        if e2 >= dy:
            if x0 == x1:
                break
            error = error + dy
            x0 = x0 + sx
        if e2 <= dx:
            if y0 == y1:
                break
            error = error + dx
            y0 = y0 + sy

    return (np.array(x_vec, dtype=int), np.array(y_vec, dtype=int))


def line(x0, y0, x1, y1):
    dx = x1 - x0
    dy = y1 - y0
    num = abs(dx) + abs(dy)
    y = np.round(np.linspace(y0, y1, num))
    x = np.round(np.linspace(x0, x1, num))
    points = np.unique(np.stack([x, y], axis=1), axis=0)
    return (points[:, 0].astype(int), points[:, 1].astype(int))
