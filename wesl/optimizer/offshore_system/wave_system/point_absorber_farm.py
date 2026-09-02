import numpy as np
from scipy.interpolate import RegularGridInterpolator
import csv
from pathlib import Path
import matplotlib.pyplot as plt

# class wave_farm ():
#     def __init__(self, fig, ax, wave_site, x_coordinates, y_coordinates, grid_x, grid_y,angles, resolution = 20):
#         self.wave_site = wave_site
#         self.x = x_coordinates
#         self.y = y_coordinates
#        = resolution
#         self.angles = angles
#         self.fig = fig  
#         self.ax = ax

#     def plot(self, base_x, base_y):
#         self.ax.scatter(self.x+base_x,self.y+base_y, label = 'Wec Device', c = 'red')
#         self.ax.set_xlabel('x [$m$]')
#         self.ax.set_ylabel('y [$m$]')   
#         self.ax.legend()
#         self.ax.axis('equal')
#         self.ax.grid(True)


#     def AEP(self):
#         print("entered AEP")
#         heights = np.array(range(1,20))
#         periods = np.array(range(1,20))
#         probabilities = self.wave_site.probability_triples(bucket_period = periods,bucket_height = heights)
#         major_probabilities = []
#         triples = []
#         total_AEP = 0
#         for dir_i in range(len(probabilities)):
#             for height_j in range(len(probabilities[dir_i])):
#                 for period_k in range(len(probabilities[dir_i][height_j])):
#                     if probabilities[dir_i][height_j][period_k] > 1e-3:
#                         # major_probabilities.append(probabilities[dir_i][height_j][period_k])
#                         wave_sim = snl_swan(oswec_x_coords=self.x, oswec_angles=self.angles, oswec_y_coords=self.y,grid_size_x=550, grid_size_y=550,
#                                             significant_wave_height=float(heights[height_j]), wave_direction=float(dir_i*60), wave_period=float(periods[period_k]), resolution=20)
#                         wave_sim.create_input_file()
#                         wave_sim.run()
#                         scenario_aep = wave_sim.calculate_total_power() 
#                         print(scenario_aep)
#                         total_AEP += scenario_aep * 8760.0 * probabilities[dir_i][height_j][period_k] / 10e9
#                         print(f"Wave_farm AEP so far: {total_AEP} GwH")
#         return total_AEP


class point_absorber(object):

    def __init__(self):
        self.space = np.array([])

        lookup_table_file_count = 16

        periods = 14
        heights = 16

        self.heights_grid = np.linspace(1,9,heights, endpoint=False)
        self.periods_grid = np.linspace(5,19,periods, endpoint=False)


        self.offset_horizontal = 60 # the distance of the wec device from the leftmost boundary in the lookup table simulations
        self.offset_vertical = 140 # the distance of the wec device from the topmost boundary in the lookup table simulations

        path = Path(__file__).parent

        data = []
        for i in range(lookup_table_file_count):
            file = np.load(f"{path}/wave_lookup/wave_lookup{i}.npy")
            data.append(file)

        data = np.array(data)

        # data = np.load(f"{path}/wave_lookup.npy")
        data = np.reshape(data, (periods, heights, 562284))

        # image = np.reshape(data[0,0,1:], (281,2001))

        self.wake = []

        self.space = data
        
        linear_interp_grid = (self.periods_grid, self.heights_grid)
        self.interpolator = RegularGridInterpolator(linear_interp_grid, self.space, method='linear', bounds_error=False, fill_value=None)


    def wakes(self, height, period):
        coordinates = (period, height)
        
        row = self.interpolator(coordinates)

        power = row[2]
        # print(row[0:3])
        wakes = row[3:]

        wakes = np.reshape(wakes, (281,2001))

        self.wake = wakes
        
        wakes = wakes[:,self.offset_horizontal:]

        # return the wake deficit
        wakes = wakes - wakes[0,0]
        
        power = np.array(power, dtype=float)
        wakes = np.array(wakes, dtype=float)
        
        return power, wakes
    
    # due to swan's precision threshold, height values can be ~1-2% off from exact value
    # use the swan value for consitency
    def height(self, height, period):
        coordinates = (period, height)
        
        row = self.interpolator(coordinates)

        power = row[0]
        wakes = row[3:]

        wakes = np.reshape(wakes, (281,2001))
        
        return(wakes[0,0])
        
    
    def sample(self,i,j):
        if (i < 0 or j < 0 or i>=len(self.wake) or j >= len(self.wake[i]) ):
            return -10
        return round(self.wake[i,j], 3)



class surrogate_swan():
    def __init__(self,x_coords, y_coords, grid_x, grid_y):
        self.coords = np.array(list(zip(x_coords,y_coords)))
        self.coords = sorted(self.coords, key=lambda x: x[1])
        self.grid_x = grid_x
        self.grid_y = grid_y

        self.lookup = point_absorber()
        self.buffer_x = 120
        self.buffer_y = 120
        self.power = 0
        
    # def power():
    # def plot()
    def evaluate(self, wave_direction=0, wave_height=1, wave_period=5, plot=True):
        # print(wave_height)
        self.power = 0
        wave_height = self.lookup.height(wave_height,wave_period)
        grid = np.full((self.grid_x, self.grid_y), float(wave_height))

        # wecs = sorted(self.coords)
        wecs = np.copy(self.coords)

        buffer = 100

        theta = np.radians(wave_direction)

        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        # Forward rotation matrix
        R = np.array([
            [cos_t, -sin_t],
            [sin_t,  cos_t]
        ])
        
        
        for i in range(len(wecs)):
            wecs[i] = np.round(np.matmul(R,wecs[i]))


        #try to keep all wecs within the top and left boundaries
        
        max_x = max(wecs[:,0])
        min_x = min(wecs[:,0])
        max_y = max(wecs[:,1])
        min_y = min(wecs[:,1])


        if min_x < 100:
            wecs[:,0] += -min_x + buffer
        if min_y < 100:
            wecs[:,1] += -min_y + buffer

        if(max_x > self.grid_x):
            raise ValueError(f"At least 1 wec position exceeds vertical (x) grid limit under direction {wave_direction}")
        if(max_y > self.grid_y):
            raise ValueError(f"At least 1 wec position exceeds horizontal (y) grid limit under direction {wave_direction}")
        
        for iteration, wec in enumerate(wecs):

            if(iteration > 0):
                height = np.mean(grid[max(0,wec[0]-10):min(wec[0]+10,len(grid[0])), wec[1]])
            else:
                height = wave_height
            power, wake = self.lookup.wakes(height, wave_period)
            self.power += power
                                            
            x_len, y_len = np.shape(wake)

            y_start = max(0, -wec[1])
            y_end = (self.grid_y - wec[1])
            
            centered_vertical = wec[0] - 140
            
            x_start = max(0, -centered_vertical)
            x_end = (self.grid_x - centered_vertical)

            # print(wec[1])
            # print(wec[1]+y_len)
            grid[max(0,centered_vertical):centered_vertical+x_len, (wec[1]):(wec[1])+y_len] = grid[max(0,centered_vertical):centered_vertical+x_len, wec[1]:wec[1]+y_len] + wake[x_start:x_end,y_start:y_end]
        
        
        self.wake = grid

        if(plot):
            plt.figure()
            plt.imshow(grid)

            plt.colorbar(label='Significant Wave Height (m)') 
            plt.title('Surrogate Wave Wakes')
            plt.xlabel(' X (m)')
            plt.ylabel(' Y (m)')

            plt.show()

    def get_last_power(self):
        return self.power

    def get_last_wakes(self):
        return self.wake

    # def AEP(self, wave_site)
        

# class wec_farm(x_coords,y_coords, wave_site):

class point_absorber_farm():
    def __init__(self, wave_site):
        self.wave_site = wave_site()
    def aep(self, x, y):
        # THESE ARE SWAPPED BECAUSE IN HERE, X IS VERTICAL AND Y IS HORIZONTAL BUT IN OPENMDAO IT IS THE OPPOSITE
        # WILL BE UPDATED
        surrogate = surrogate_swan(x_coords=y, y_coords=x, grid_x= np.sqrt(max(y)**2 + max(x)**2), grid_y=np.sqrt(max(y)**2 + max(x)**2))

        aep = 0

        height_buckets = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        period_buckets = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        directions_num = self.wave_site.direction_num
        probabilities = self.wave_site.probability_triples(height_buckets, period_buckets)

        for i in range(directions_num):
            for j in range(len(period_buckets)-1):
                for k in range(len(height_buckets)-1):
                    height = (height_buckets[k] + height_buckets[k]+1) / 2
                    period = (period_buckets[j] + period_buckets[j]+1) / 2
                    direction = i * 360 / directions_num 
                    surrogate.evaluate(wave_direction = direction, wave_height = height, wave_period = period, plot = False)
                    aep += surrogate.get_power() * probabilities[i,j,k]
        return aep


if __name__ == "__main__": 
    from wesl.optimizer.offshore_system.wave_system.snl_swan_wrapper import snl_swan


#     path = Path(__file__).parent
#     print(f"{path}/wave_lookup.npy")
#     data = np.load(f"{path}/wave_lookup.npy")
# # wesl/optimizer/offshore_system/wave_system/wave_lookup.npy
#     data_count = len(data)

#     data_split = []
#     data_split_file_count = 16

#     for i in range(data_split_file_count):
#         print(np.ceil(i*data_count/data_split_file_count).astype(int),np.ceil((i+1)*data_count/data_split_file_count).astype(int), np.shape(data))
#         data_split.append(data[np.ceil(i*data_count/data_split_file_count).astype(int):np.ceil((i+1)*data_count/data_split_file_count).astype(int)])
#         print(np.shape(data[np.ceil(i*data_count/data_split_file_count).astype(int):np.ceil((i+1)*data_count/data_split_file_count).astype(int)]))

#     data_split = np.array(data_split)

#     for i in range(data_split_file_count):
#         np.save(f"{path}/wave_lookup/wave_lookup{i}.npy", data_split[i])
        
    
    x_coords = [101, 201, 301, 401, 501,
                101, 201, 301, 401, 501,
                101, 201, 301, 401, 501]

    y_coords = [101, 101, 101, 101, 101,
                251, 251, 251, 251, 251,
                401, 401, 401, 401, 401]

    period = 14
    wave_height = 2.5

    # grid size of surrogate_swan
    horizontal_grid = 1000
    vertical_grid = 600

    surrogate = surrogate_swan(x_coords=x_coords,y_coords=y_coords, grid_y=horizontal_grid, grid_x=vertical_grid)

    swan = snl_swan(wec_type=0, wec_x_coords=x_coords, wec_y_coords=y_coords, resolution= 10, wave_direction=90, significant_wave_height = wave_height, wave_period = period, grid_size_x=horizontal_grid, grid_size_y=vertical_grid)
    swan.create_input_file()
    swan.run()
    print(f"Power of Farm in Swan: {swan.calculate_total_power() / 1000} KW")
    swan.plot_wakes()

    surrogate.evaluate(wave_height=wave_height, wave_period = period)

    print(f"Power of Farm in Surrogate: {surrogate.get_last_power()/1000} KW")

