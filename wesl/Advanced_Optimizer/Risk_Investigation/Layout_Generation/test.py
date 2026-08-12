import numpy as np
from shapely.geometry import Point, Polygon
from shapely import affinity

def generate_tiling_grid(polygon, spacing_x, spacing_y, angle_deg, offset=(0, 0)):
    """
    Generates a turbine grid based on tiling.
    
    Unlike a simple grid, this allows grid rotation INDEPENDENT of the shape,
    aligning with nautical patterns (e.g., 1x1 nm).
    """
    # 1. Get the expanded bounding box to cover the rotation
    xmin, ymin, xmax, ymax = polygon.bounds
    diag = np.sqrt((xmax - xmin)**2 + (ymax - ymin)**2)
    center_x, center_y = polygon.centroid.x, polygon.centroid.y
    
    # 2. Generate points on a local (cartesian) grid
    num_x = int(diag / spacing_x) + 2
    num_y = int(diag / spacing_y) + 2
    
    xs = (np.arange(num_x) - num_x // 2) * spacing_x + offset[0]
    ys = (np.arange(num_y) - num_y // 2) * spacing_y + offset[1]
    xv, yv = np.meshgrid(xs, ys)
    points = np.vstack([xv.ravel(), yv.ravel()]).T
    
    # 3. Apply Rotation (Alignment Constraint)
    angle_rad = np.radians(angle_deg)
    rot_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad),  np.cos(angle_rad)]
    ])
    rotated_points = points @ rot_matrix.T
    
    # 4. Translate to the center of the polygon
    final_points = rotated_points + np.array([center_x, center_y])
    
    # 5. Filter points inside the polygon
    mask = [polygon.contains(Point(p)) for p in final_points]
    return final_points[mask]

# Example usage to be integrated into build_scaled_farm:
# pts = generate_tiling_grid(shape, spacing_m, spacing_m, alignment_angle)