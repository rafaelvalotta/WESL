import numpy as np
import openmdao as om
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from wesl.optimizer.offshore_system.wave_system.point_absorber_farm import point_absorber_farm


class PointAbsorberFarm(om.ExplicitComponent):

    def initialize(self):
        self.options.declare("layout_coordinates", 
                             types=np.ndarray, 
                             desc = "Point Absorber layout coordinates")
        self.options.declare("sim_res", 
                             desc="Surrogate Model")

        self.options.declare("n_wecs", 
                             types = int,
                             desc="number of wec devices")
        
        self.options.declare('boundary', types=np.ndarray)
        self.options.declare('spacing_diameter', default=100, types=(float, int)) # upgrade here for the spacing
        self.options.declare("layout_coordinates", types=np.ndarray)
        self.options.declare("x_grid", types = np.ndarray)
        self.options.declare("AEP_init", types=float)



    def setup(self):
        # Setting layout coordinates as inputs       
        self.add_input('x', np.zeros(len(self.options["layout_coordinates"][0])))  # X-Layout Coordinates
        self.add_input('y', np.zeros(len(self.options["layout_coordinates"][1])))  # Y-Layout Coordinates


        # Setting AEP as output
        self.add_output('AEP', val=0.0)

        self.n_wec = len(self.options["layout_coordinates"][0])
        xl, xu, yl, yu = self.options["plot_lim"]

        x_coordinates = self.options["layout_coordinates"][0]
        y_coordinates = self.options["layout_coordinates"][1]
        boundary =  self.options["boundary"]
        lon_grid_fine = self.options["lon_grid_fine"]
        lat_grid_fine = self.options["lat_grid_fine"]
        interpolated_elevation = self.options["interpolated_elevation"]

        self.iteration = 0
        self.circles = []
        self.turbine_scatter = None  
        self.cableA = None
        self.cableB = None

        self.fig, self.ax = plt.subplots()
        

        # Defines the water depth map
        plt.pcolormesh(lon_grid_fine, 
                    lat_grid_fine, 
                    interpolated_elevation, 
                    cmap='Blues_r', 
                    shading='auto', 
                    vmin=-120, 
                    vmax=-60)

        plt.colorbar(label="Water Depth (m)")
        plt.plot(boundary[:, 0], 
                boundary[:, 1], 
                label='Boundary', 
                c='black', 
                linestyle = '--')
        plt.tight_layout()
        # plt.ion()
        self.ax.scatter(x_coordinates,
                        y_coordinates, 
                        c='orange', 
                        marker = '.', 
                        s=8, 
                        label='Initial Layout')
        self.text_box = self.ax.text(0.01, 
                                    0.99, 
                                    '', 
                                    transform=self.ax.transAxes, 
                                    verticalalignment='top', 
                                    fontsize=10, 
                                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        # self.text_box.set_text(f"Iteration: {self.iteration}\nAEP Improvement: {-aep} %")
        # self.text_box.set_text(f"AEP Improvement: {-aep} %")


        self.ax.set_xlabel('X [m]')
        self.ax.set_ylabel('Y [m]')
        # self.ax.set_xlim(360000, 390000)
        # self.ax.set_ylim(4.53E6, 4.56E6)

        # self.ax.set_xlim(300000, 350000)
        # self.ax.set_ylim(4.54E6, 4.58E6)
        self.ax.set_xlim(xl, xu)
        self.ax.set_ylim(yl, yu)

        print('done')

        

    def compute(self, inputs, outputs):


        print('Entered compute wec AEP compute')
        outputs['AEP'] = -self.options["sim_res"](inputs['x'], inputs['y']).aep()


        x = inputs['x']
        y = inputs['y']


        # aep = inputs['AEP'].item()
        aep = outputs['AEP'].item()
        aep_init = self.options["AEP_init"]

        x_coordinates = self.options["layout_coordinates"][0]
        y_coordinates = self.options["layout_coordinates"][1]

        spacing_radius = self.options['spacing_diameter'] / 2

        if self.turbine_scatter is not None:
            self.turbine_scatter.remove()

        # Remove old circles
        for circ in self.circles:
            circ.remove()
        self.circles.clear()

        self.turbine_scatter = self.ax.scatter(x,
                                               y,
                                               marker = '2', 
                                               c='black', 
                                               label='Current Design')

        # Draw new wec circles
        for xi, yi in zip(x, y):
            circ = Circle((xi, yi), 10, edgecolor='gray',
                          linestyle='--', facecolor='none', linewidth=1)
            self.ax.add_patch(circ)
            self.circles.append(circ)

        plt.draw()
        plt.pause(0.001) 
        # self.ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, fontsize=10)
        # Rebuild legend without duplicates
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))  # removes duplicates based on label
        self.ax.legend(by_label.values(), by_label.keys(),
                    loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, fontsize=10)

        self.text_box.set_text(
            f"Iteration: {self.iteration}\nAEP Improvement: {((aep / aep_init) - 1) * 100:.3f} %"
        )


        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        # plt.show()
        self.iteration += 1

