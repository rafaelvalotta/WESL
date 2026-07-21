"""
Generates the candidate layouts for the adjacent SouthCoast Wind lease,
one per candidate turbine, filling the real lease polygon with a regular
grid spaced at 7 rotor diameters (7D) -- methodology Section 3.4 / 5.1.6.

Algorithm:
  1. Generate candidates on a regular grid covering the polygon's bounding
     box, spaced at 7x the chosen turbine's diameter.
  2. Filter with polygon.contains(Point(x, y)), keeping only points inside
     the real boundary.
  3. Turbine count and installed capacity are derived outputs, not chosen
     parameters.

Runs once, in preprocessing (not inside the 216-combination loop -- the
layout only depends on which turbine is being tested, not on year or
wake model).
"""

import numpy as np
from shapely.geometry import Point, Polygon

# Real SouthCoast lease polygon (UTM zone 19N, same CRS as Vineyard and
# Revolution in site_us.yaml)

SOUTHCOAST_X = [390970.126, 391008.248, 389681.412, 389608.225, 388508.51, 388375.705, 387279.262, 387263.441, 386222.785, 386207.706, 384875.562, 384860.308, 383761.546, 383686.592, 382646.291, 382453.83, 381353.738, 381337.103, 380179.973, 380160.378, 378889.36, 378870.501, 377656.114, 377635.162, 376420.547, 376400.351, 375302.094, 375224.877, 374009.845, 373873.619, 372889.657, 372812.054, 371598.592, 371459.946, 370418.799, 370396.638, 369181.759, 369161.425, 368175.748, 368058.829, 366886.312, 366796.424, 365623.71, 365565.608, 364460.382, 364372.606, 363233.341, 363211.839, 362141.992, 362048.142, 360804.962, 360846.742, 370340.687, 370416.52, 372798.962, 372855.211, 374063.488, 374083.84, 375256.828, 375278.144, 376348.55, 376402.476, 377576.784, 377629.95, 378768.436, 378858.153, 380030.937, 380050.901, 381189.55, 381242.16, 382449.044, 382433.478, 383571.213, 383624.533, 384797.621, 384850.208, 385953.63, 386041.02, 387144.252, 387196.988, 388367.904, 388489.854, 389627.159, 389610.604, 390816.092, 390868.734, 392074.519, 392092.518, 393228.735, 393246.051, 394485.358, 394433.634, 395603.9, 395586.908, 396824.853, 396774.215, 397978.023, 398028.513, 399095.466, 399145.309, 394328.699, 394396.806, 390970.126]

SOUTHCOAST_Y = [4531056.793, 4529734.31, 4529754.06, 4528720.44, 4528507.039, 4527359.153, 4527375.836, 4526341.014, 4526241.979, 4525264.499, 4524997.579, 4524019.931, 4523922.101, 4522830.175, 4522789.101, 4521583.789, 4521428.737, 4520393.07, 4520296.631, 4519088.15, 4519108.866, 4517957.749, 4517920.165, 4516653.72, 4516616.324, 4515407.233, 4515425.657, 4514274.932, 4514237.905, 4513030.387, 4512989.593, 4511838.502, 4511917.059, 4510593.957, 4510612.087, 4509344.371, 4509365.706, 4508213.05, 4508115.19, 4507130.66, 4507117.256, 4505951.017, 4505937.803, 4504633.377, 4504550.393, 4503521.174, 4503473.335, 4502305.275, 4502290.676, 4500951.891, 4500906.311, 4497604.698, 4497605.193, 4494026.612, 4493985.559, 4495257.456, 4495236.927, 4496440.43, 4496386.292, 4497658.362, 4497674.887, 4498843.046, 4498892.39, 4500025.994, 4500007.348, 4501277.665, 4501258.645, 4502495.516, 4502511.586, 4503644.647, 4503625.449, 4504820.821, 4504802.894, 4506004.287, 4506054.68, 4507221.541, 4507238.839, 4508439.314, 4508456.779, 4509657.59, 4509605.537, 4510839.443, 4510856.741, 4512058.195, 4512040.33, 4513274.873, 4513291.513, 4514526.352, 4514509.876, 4515710.217, 4515692.441, 4516893.571, 4516876.961, 4518111.682, 4518025.735, 4519329.289, 4519278.291, 4520443.105, 4520462.733, 4521593.098, 4521591.898, 4531146.829, 4531056.793]

SOUTHCOAST_POLYGON = Polygon(zip(SOUTHCOAST_X, SOUTHCOAST_Y))

CANDIDATE_TURBINES = {
    "option1_haliadex_13mw": dict(diameter=220.0, power_kw=13000),
    "option2_haliadex_250": dict(diameter=250.0, power_kw=15500),
    "option3_gwh300_22mw": dict(diameter=300.0, power_kw=22000),
}


def generate_7d_grid_layout(polygon, rotor_diameter):
    """
    Fills 'polygon' with a regular grid spaced at 7*rotor_diameter,
    keeping only points inside the real polygon.

    Returns (x, y, n_turbines) -- position arrays and the count (a
    derived output, not a chosen parameter).
    """
    spacing = 7 * rotor_diameter

    xmin, ymin, xmax, ymax = polygon.bounds
    xs = np.arange(xmin, xmax + spacing, spacing)
    ys = np.arange(ymin, ymax + spacing, spacing)

    x_valid, y_valid = [], []
    for x in xs:
        for y in ys:
            if polygon.contains(Point(x, y)):
                x_valid.append(x)
                y_valid.append(y)

    x_valid = np.array(x_valid)
    y_valid = np.array(y_valid)
    return x_valid, y_valid, len(x_valid)


def generate_all_layouts(polygon=SOUTHCOAST_POLYGON, turbines=CANDIDATE_TURBINES, verbose=True):
    """
    Returns {turbine_key: dict(x=..., y=..., n_turbines=..., capacity_mw=...)}
    -- one candidate layout per turbine, generated once.
    """
    layouts = {}

    area_km2 = polygon.area / 1e6

    for key, spec in turbines.items():
        x, y, n = generate_7d_grid_layout(polygon, spec["diameter"])
        capacity_mw = n * spec["power_kw"] / 1000.0

        layouts[key] = dict(x=x, y=y, n_turbines=n, capacity_mw=capacity_mw, diameter=spec["diameter"])

        if verbose:
            print(f"[{key}] D={spec['diameter']:.0f}m, 7D spacing={7*spec['diameter']:.0f}m "
                  f"-> {n} turbines, capacity={capacity_mw:.0f} MW (lease area: {area_km2:.1f} km2)")

    return layouts


if __name__ == "__main__":
    print(f"SouthCoast polygon: area = {SOUTHCOAST_POLYGON.area/1e6:.1f} km2, "
          f"bounding box = {SOUTHCOAST_POLYGON.bounds}\n")

    layouts = generate_all_layouts()

    print("\nSummary:")
    for key, data in layouts.items():
        print(f"  {key}: {data['n_turbines']} turbines, {data['capacity_mw']:.0f} MW installed")
