import subprocess
import os
from matplotlib import pyplot as plt
from scipy.io import loadmat
import math
import numpy as np
from pathlib import Path

#https://www.researchgate.net/publication/288496684_Development_Verification_and_Application_of_the_SNL-SWAN_Open_Source_Wave_Farm_Code

class snl_swan():
    """
    Python Wrapper for SNL-SWAN, swan.exe must be compiled for your own operating system
    Instructions for Source code: https://sandialabs.github.io/SNL-SWAN/getting_started.html#downloading-snl-swan
    After git cloning, the instructions for building the library are inside the folder
    This object relies on input files in the same directory
    Running multiple instances in parallel may cause a race condition

    Note: not all swan capabilities are exposed by this wrapper

    Parameters:
    --------------------------------------------------
    wec_type              Informs whether wec to be simulated is rm3 point absorber (0) or rm5 oswec (1). wec_type=0 will override oswec angles
    Oswec_angles          Angle of each oswec. Should be the same length as x and y coords
    wec_x_coords          Center x position of each wec device
    wec_x_coords          Center y position of each wec device
    grid_size_x           size of simulated x area in meters
    grid_size_y           size of simulated y area in meters
    resolution            size of each descrete segment of grid in meters
   
    """
    def __init__(self, wec_type = 0, oswec_angles = [], wec_x_coords=[], wec_y_coords=[],
                 grid_size_x=2000, grid_size_y=250, resolution = 20, significant_wave_height = 5.5,
                 wave_period = 8, wave_direction = 0, wave_spread = 45, x_vertical = True):

        if x_vertical:
            temp = wec_x_coords
            wec_x_coords = wec_y_coords 
            wec_y_coords = temp

        # oswec type: 0 is point absorber, 1 is oswec
        # point absorbers are rotationally symmetrical
        # their maximum width is always oriented towards the incoming wave
        # so point the device towards the wave direction
        self.wec_type = wec_type
        if(wec_type==0):
            oswec_angles = np.zeros(len(wec_x_coords))
            # meters
            self.oswec_width = 20
            for i in range(len(oswec_angles)):
                oswec_angles[i] = wave_direction + 90 
        elif(wec_type==1):
            self.oswec_width = 25
        else:
            raise(ValueError("wec_type must be 0 or 1"))
      
        self.path = Path(__file__).parent

        self.file_name = 'INPUT'
        self.obcase = 3
        self.wec_device_power = []
        # in degrees
        self.oswec_angles = oswec_angles

        # meters
        self.wec_x_coords = wec_x_coords
        self.wec_y_coords = wec_y_coords
        self.wec_number = len(wec_x_coords)

        # meters
        self.grid_size_x = grid_size_x
        self.grid_size_y = grid_size_y


        # this value is the spatial resolution of the simulation, 
        # at a value of 1, the grid is descretized in 1 meter segments
        # at 5, 5 meter segments, etc
        self.resolution = resolution


        self.significant_wave_height = significant_wave_height
        self.wave_period = wave_period

        self.wave_direction = wave_direction
        #coefficient of directional spreading
        self.wave_spread = wave_spread

    def create_input_file(self):
        with open(f"{self.path}/{self.file_name}", 'w') as file:
            file.write("Project 'SAMPLE' 'TEST'\n")
            file.write("SET CARTESIAN\n")
            file.write("SET inrhog = 1\n")
            file.write(f"SET obcase = {self.obcase}\n")
            file.write("MODE STAT TWOD\n")
            # file.write("GEN3 KOMEN 0 3.02e-3\n")
            file.write("COORD CARTESIAN\n")
            file.write(f"CGRID REG 0.0 0.0 0.0 {self.grid_size_x} {self.grid_size_y} {self.grid_size_x // self.resolution} {self.grid_size_y // self.resolution} CIRCLE 180 .05 .5 25\n")
            file.write("INPGRID BOTTOM REG 0.0 0.0 0.0 100 100 10 10\n")
            file.write("READINP BOTTOM 1.0000 'Bathymetry.bot' 3 0 FREE\n")
            file.write("BOUND SHAPESPEC JONSWAP 3.3 PEAK DSPR POWER\n")
            file.write(f"BOUNDSPEC SIDE N CON PAR {self.significant_wave_height} {self.wave_period} {self.wave_direction} {self.wave_spread}\n")
            file.write(f"BOUNDSPEC SIDE W CON PAR {self.significant_wave_height} {self.wave_period} {self.wave_direction} {self.wave_spread}\n")
            file.write(f"BOUNDSPEC SIDE S CON PAR {self.significant_wave_height} {self.wave_period} {self.wave_direction} {self.wave_spread}\n")
            file.write(f"BOUNDSPEC SIDE E CON PAR {self.significant_wave_height} {self.wave_period} {self.wave_direction} {self.wave_spread}\n")

            file.write("BREAKING\n")
            file.write("FRICTION\n")
            file.write("OFF QUADRUPL\n")

            for i in range(self.wec_number):

                x1 = self.wec_x_coords[i] - math.cos(math.radians(self.oswec_angles[i])) * self.oswec_width / 2
                y1 = self.wec_y_coords[i] - math.sin(math.radians(self.oswec_angles[i])) * self.oswec_width / 2
                x2 = self.wec_x_coords[i] + math.cos(math.radians(self.oswec_angles[i])) * self.oswec_width / 2
                y2 = self.wec_y_coords[i] + math.sin(math.radians(self.oswec_angles[i])) * self.oswec_width / 2
                file.write(f"OBSTACLE TRANS 0.3 REFL 0.00 LINE {x1:.3f} {y1:.3f} {x2:.3f} {y2:.3f}\n")

            file.write ("TABLE 'COMPGRID' NOHEAD 'SWANOUT.DAT' XP YP HSIGN DIR RTP DEP\n")
            file.write("BLOCK 'COMPGRID' NOHEAD 'SWANOUT.mat' LAY 3 HSIGN DIR RTP TDIR\n")
            file.write("COMPUTE\n")
            file.write("STOP")

    def run(self):

        # selecting the right power matrix for simulation
        if(self.wec_type == 0):
            subprocess.run(["mv", "Power_rm3.txt", "Power.txt"], cwd = self.path)
        else:
            subprocess.run(["mv", "Power_rm5.txt", "Power.txt"], cwd = self.path)

        # run compiled swan program
        subprocess.run([f"./swan.exe", "-input", "INPUT", "-omp", "6"],cwd=self.path, stdout=subprocess.DEVNULL)

        if(self.wec_type == 0):
            subprocess.run(["mv", "Power.txt", "Power_rm3.txt"], cwd = self.path)
        elif(self.wec_type == 1):
            subprocess.run(["mv", "Power.txt", "Power_rm5.txt"], cwd = self.path)

        # read output power file
        with open(f"{self.path}/POWER_ABS.OUT", "r") as power_file:
            lines = power_file.readlines()
            lines.reverse()
            for i in range(self.wec_number):
                words = lines[i].split()
                self.wec_device_power.append(float(words[6]))
            self.wec_device_power.reverse()


    def plot_wakes(self):
        """
        Will show simple matrix of last swam simulation
        """
        matrix = loadmat(f'{self.path}/SWANOUT.mat')

        plt.matshow(matrix['Hsig']) 
        plt.colorbar(label='Significant Wave Height (m)') 
        plt.title('Swan Wave Wakes')
        plt.xlabel(f'X resolution ({self.resolution} m)')
        plt.ylabel(f'Y resolution ({self.resolution} m)')

        plt.show()

        return matrix

    def get_wakes(self):
        matrix = loadmat(f'{self.path}/SWANOUT.mat')
        return matrix
    

    def calculate_total_power(self):
        return sum(self.wec_device_power)

if __name__ == "__main__":
    angles = [0,]# 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    x_coords = [20,]# 50, 80, 110, 140, 170, 200, 230, 260, 290, 320]

    y_coords = [125,]# 125, 125, 125, 125, 125, 125 ,125, 125 ,125 ,125]

    sim = snl_swan(wec_type= 0, oswec_angles=angles, wec_x_coords=x_coords, wec_y_coords=y_coords, resolution= 5, wave_direction=0, significant_wave_height = 5, wave_period=14)

    sim.create_input_file()
    sim.run()

    sim.plot_wakes()

    # column_power = [0,0,0,0,0]
    # column_percentage = [1,1,1,1,1]

    # for i in range(5):
    #    for j in range(5):
    #        column_power[i] += sim.wec_device_power[i*5 + j]
    #    column_percentage[i] = column_power[i] / column_power[0]

    # print(column_power)
    # print(column_percentage)

    # power = (sim.calculate_total_power())
    # aep = power * 8760 / 10**9 # convert to gigawatts hours
    # print(aep)

    # sim.plot_wakes()

