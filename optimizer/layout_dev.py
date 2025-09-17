import numpy as np
import xarray as xr
from shapely.geometry import Polygon, Point
from windFarms_windTurbines import *
import random
from scipy.spatial import KDTree


def compute_number_of_turbines(area=267, rated_power=13, capacity_density=3):
    """
    Compute the number of turbines based on the site area, turbine rated power,
    and desired capacity density (MW/km^2).
    """
    num_turbines = (capacity_density * area) / rated_power
    return int(round(num_turbines))  # or round appropriately

### The following function must be improved so that when n_wt is offered, it must guarantee that 
# all the n_wt turbines are placed within the boundaries which means I must implement the spacing strategy, also figure out a way
# to adjust the code so that capacity density becomes a parameter so that it can better layout the grid pos of turbines 
# (refer to aep_n_trubines.py to see how to implement it) !!!! ATTENTION !!!!
def grid_WTposition_generator(boundary_points, windTurbine, n_wt=None, spacing=10):
    """
    Generates a grid of turbine positions within a given boundary.
    
    Turbine centers are placed on a grid in order from top to bottom, left to right.
    The turbine center is placed only if it is strictly within the boundary and
    the grid spacing is set to 2 times the rotor diameter.
    
    Parameters:
      n_wt (int or None): Number of turbines to place. If None, all valid positions
                          in the grid (given the spacing and boundary restrictions)
                          will be populated.
      boundary_points (list): List of (x, y) tuples defining the boundary polygon.
      windTurbine: An object with a .diameter() method that returns the rotor diameter.
      spacing: how many diameter spacing in diameters (spacing * D)
    
    Returns:
      tuple: Two tuples (wt_x, wt_y) containing the x and y coordinates of turbine positions.
    """
    # Get the rotor diameter and set grid spacing to spacing * rotor diameter.
    rotor_diameter = windTurbine.diameter()
    spacing = spacing * rotor_diameter
    
    # Create the boundary polygon.
    boundary_polygon = Polygon(boundary_points)
    
    # Obtain the bounding box of the polygon.
    min_x, min_y, max_x, max_y = boundary_polygon.bounds
    
    # To ensure turbines do not touch the boundary, define a margin.
    # Here we use rotor_diameter as a simple margin.
    x_start = min_x + rotor_diameter
    x_end = max_x - rotor_diameter
    y_start = max_y - rotor_diameter  # starting from the top
    y_end = min_y + rotor_diameter
    
    # Generate grid coordinates.
    x_coords = []
    x = x_start
    while x <= x_end:
        x_coords.append(x)
        x += spacing

    y_coords = []
    y = y_start
    while y >= y_end:
        y_coords.append(y)
        y -= spacing

    # Populate the grid in row-major order (top to bottom, left to right).
    turbine_positions = []
    for y in y_coords:
        for x in x_coords:
            point = Point(x, y)
            # Check that the turbine point is strictly inside the boundary.
            if boundary_polygon.contains(point):
                turbine_positions.append((x, y))
                # If a fixed number of turbines is requested, stop when reached.
                if n_wt is not None and len(turbine_positions) >= n_wt:
                    break
        if n_wt is not None and len(turbine_positions) >= n_wt:
            break

    if not turbine_positions:
        print("\033[91mError: No turbines were able to be positioned within the given boundary!\033[0m")
        exit(1)
    
    # Separate x and y coordinates for the return value.
    print(f"Total Turbine Deployed: {len(turbine_positions)}")
    wt_x, wt_y = zip(*turbine_positions)
    #print("Total Turbones = "+ str(len(wt_x)))
    wt_x, wt_y = np.array(wt_x), np.array(wt_y)
    return wt_x, wt_y


def random_WTposition_generator(n_wt, boundary_points, windTurbine, spacing=10):
    wt_diameter = windTurbine.diameter()
    turbines_spacing = spacing * wt_diameter  # Adjusted for clear turbine safety spacing

    # Create the boundary polygon
    boundary_polygon = Polygon(boundary_points)

    # Get bounds of the polygon to limit random point generation
    min_x, min_y, max_x, max_y = boundary_polygon.bounds

    # List to store turbine positions as Point objects
    turbine_positions = []

    # Build an initial KDTree with an empty dataset (will be updated)
    kd_tree = None

    # Attempt counter to prevent infinite loop
    max_attempts_per_turbine = 10000
    attempt_counter = 0

    # Generate wind turbine positions within the polygon with proper spacing
    while len(turbine_positions) < n_wt:
        if attempt_counter >= max_attempts_per_turbine:
            print(f"\033[93mWARNING: The boundary is too small for the amount of turbines wanted to be placed given the turbine spacing safety protocol. Thus only {len(turbine_positions)} can be placed!\033[0m")
            break

        # Generate random coordinates within the bounding box
        x = random.uniform(min_x + turbines_spacing, max_x - turbines_spacing)
        y = random.uniform(min_y + turbines_spacing, max_y - turbines_spacing)
        point = Point(x, y)

        # Ensure the point is within the polygon
        if boundary_polygon.contains(point):
            if kd_tree is not None:
                # Query the KDTree to find the nearest neighbor
                distance, _ = kd_tree.query([x, y], k=1)

                # Check if the distance is greater than or equal to the required spacing
                if distance >= turbines_spacing:
                    turbine_positions.append((x, y))
                    kd_tree = KDTree(turbine_positions)  # Update the KDTree with the new point
                    attempt_counter = 0  # Reset attempt counter after successful placement
                else:
                    attempt_counter += 1
            else:
                # If no KDTree exists yet, simply add the first point
                turbine_positions.append((x, y))
                kd_tree = KDTree(turbine_positions)  # Create KDTree with the first point
                attempt_counter = 0  # Reset attempt counter after placing the first turbine

    # Separate x and y coordinates for return
    if turbine_positions:
        wt_x, wt_y = zip(*turbine_positions)
    else:
        # Print an error message in red and exit the program
        print("\033[91mError: No turbines were able to be positioned!\033[0m")
        exit(1)  # Exit the script with an error status (1)

    return wt_x, wt_y

## Optimization Layout Below

