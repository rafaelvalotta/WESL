import numpy as np
from scipy.interpolate import RegularGridInterpolator
import csv
from pathlib import Path
from scipy.stats import weibull_min

class FOWT_Spar(object):
    """ FOWT with spar-type floater, assuming station keeping is done
    by three mooring lines configured symmetricaly, i.e., the mooring
    lines are of same length, the water depth is uniform and the
    projections of adjacent moorinng lines are seperated by 120 degree.

    """
    def __init__(self):
        csv_file = []
        self.space = np.array([])

        wind_directions = 19
        wave_directions = 7
        wind_speeds = 10
        wave_heights = 4

        # add better resolution later
        self.wind_directions_grid = np.linspace(0,360,wind_directions, endpoint=True)
        self.wave_directions_grid = np.linspace(0,360,wave_directions, endpoint=True)
        self.wind_speed_grid = np.linspace(4,22,wind_speeds)
        self.wave_height_grid = np.arange(1,8, 2)
        header = []

        # print(self.wind_directions_grid,self.wave_directions_grid,self.wind_speed_grid,self.wave_height_grid)

        path = Path(__file__).parent
        with open(f"{path}/wave_wind_dirs_fixed.csv", "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                row = [float(value) for value in row]
                csv_file.append(row)

        self.space = np.array(csv_file)
        self.lookup_table = csv_file
        self.space = self.space.reshape(wind_directions, wave_directions, wave_heights, wind_speeds, 10)

        linear_interp_grid = (self.wind_directions_grid, self.wave_directions_grid, self.wave_height_grid, self.wind_speed_grid)
        self.interpolator = RegularGridInterpolator(linear_interp_grid, self.space, method='linear')


    def solve_static_movement(self, wind_speed, wind_direction, wave_height=1, wave_direction=0):
        """
        Function to compute steady state movement of FOWT under environmental conditions
         with a given inflow wind direction.

        Parameters
        ----------
        wind_speed : int [m/s]
            Wind Speed at rotor center 
        wind_direction : float [deg]
            Inflow wind direction. Clockwise is positive
        Wave_Height : float [m]
            Significant : Height of incoming wave
        wave_direction :  float [deg]
            Inflow wave direction. Clockwise is positive
        ----------


        Note:
        openfast 0 wind direction is (1,0) on the xy plane
        rotated 90 degrees counterclockwise from pywake

        Returns
        -------
        x_r_change : float or array [m]
            Movement in x direction of rotor center.
        y_r_change : float or array [m]
            Movement in y direction of rotor center.
        z_r_change : float or array [m]
            Movement in z direction of rotor center.
        gamma_in_deg : float or array [deg]
            Pitch angle of platform in degree.
        """

        #converts wind direction from pywake to openfast lookup table
        
        wind_direction = np.fmod((wind_direction - 90), 360)
        wind_direction = np.fmod((wind_direction + 360), 360)
        # print(wind_speed)
        # this exists because sometimes wind_speed looks like [[windspeed]] due to code elsewhere. Should be fixed eventually
        while(type(wind_speed)==np.ndarray and wind_speed.shape):
            # print(wind_speed)
            wind_speed = wind_speed[0]


        # convert wind_speed into value within lookup table
        # due to wind speed thrust curve, values over ~22 make infinitesimally small changes in displacement 
        wind_speed = max(wind_speed, 4)
        wind_speed = min(wind_speed, 22)
        wind_speed = wind_speed

        # current limits of lookup table
        wave_height = min(wave_height, 7)
        wave_height = max(wave_height, 1)
        wave_height = wave_height




        coordinates = (wind_direction, wave_direction, wave_height, wind_speed)
        row = self.interpolator(coordinates)

        position = row[4:7]
        angles = row[7:]

        position = np.array(position, dtype=float)
        angles = np.array(angles, dtype=float)


        x_r_change = -position[0]
        y_r_change = -position[1]
        z_r_change = position[2]

        # tilt can be positive or negative depending on orientation in openfast
        # always use positive value for pywake, since turbines will only tilt backwards to the sky
        gamma_in_deg = np.abs(angles[0])

        return x_r_change, y_r_change, z_r_change, gamma_in_deg

#https://www.sciencedirect.com/science/article/pii/S0378383907000452
#https://cdip.ucsd.edu/m/products/rose/?stn=160p1
class wave_site():
    def __init__(self, f, a, k):
        if (len(f) != len(a) or len(f) != len(k)):
            raise ValueError("weibull parameters for waves are not equal")
        self.f = f
        self.a = a
        self.k = k
        self.weibull = []
        # for i in range(len(self.f)):
        #     self.weibull.append(weibull_min(c=self.k[i], scale = a))

    def probabilities(self, buckets):

        # probability matrix of the form [directions, wave_height]

        probabilities = []
        for i in range(len(self.f)):
            cdf_values = weibull_min.cdf(buckets, c=self.k[i], scale = self.a[i])
            bucket_prob = np.diff(cdf_values)
            probabilities.append(bucket_prob*self.f[i])
        return np.array(probabilities)



if __name__ == "__main__":
    fowt = FOWT_Spar()
    print(fowt.solve_static_movement(wind_direction=356,wind_speed=4, wave_direction = 0, wave_height = 0))


    wave_f = [8.37, 6.83, 5.38, 4.77, 4.10, 4.80,
        8.22, 13.29, 11.22, 10.14, 11.72, 11.16]
    wave_a = [3.267, 3.267, 3.267, 3.267, 3.267, 3.267, 3.267, 3.267, 3.267, 3.267, 3.267, 3.267]
    wave_k = [4, 4 ,4 ,4 ,4 ,4 ,4 ,4 ,4 ,4 ,4 ,4]

    test_wave_site = wave_site(f=wave_f / np.sum(wave_f),a=wave_a,k=wave_k)
    print(np.sum(np.array(test_wave_site.probabilities([0,1,2,3,4,5,6,7]))))
    # print(sum())