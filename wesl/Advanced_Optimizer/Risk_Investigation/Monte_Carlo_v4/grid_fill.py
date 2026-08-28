import numpy as np
from shapely.geometry import Point


def fill(polygon, diameter, spacing_d):
    spacing = spacing_d * diameter
    xmin, ymin, xmax, ymax = polygon.bounds
    xs = np.arange(xmin, xmax + spacing, spacing)
    ys = np.arange(ymin, ymax + spacing, spacing)
    pts = [(x, y) for x in xs for y in ys if polygon.contains(Point(x, y))]
    if not pts:
        return np.empty((0, 2))
    return np.array(pts)
