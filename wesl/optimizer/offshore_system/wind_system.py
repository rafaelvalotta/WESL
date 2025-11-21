import numpy as np
import openmdao.api as om
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from wesl.optimizer.interarray.farmrepo import g1

# Heuristic Wrapper Valotta Rodrigues Perez 2024 (Mauricio Souza DTU thesis 2022)
from wesl.optimizer.interarray.interface import heuristic_wrapper
from wesl.optimizer.interarray.farmrepo import g1

from py_wake.utils.gradients import autograd

plt.ion()  

class FixedBottomWindFarm(om.ExplicitComponent):

    """
    Fixed-Bottom Offshore Wind Farm System for AEP Layout Optimization

    Parameters:
    ----------------------------------------------------------------------------------
    Variable                      Description

    layout_coordinates (float):   x and y wind turbine coordinates 
    sim_res (xarray, float):      Instance of wind farm model in xarray format
    AEP (xarray, float):          Annual Energy Production in xarray format (PyWake)
    ----------------------------------------------------------------------------------

    Usage:
    ----------------------------------------------------------------------------------
    User gives wind turbine layout coordinates (x,y) and the sim_res object when setting 
    the OpenMDAO component. The x and y turbine coordinates can be user-defined, or be
    retrieved using the boundary_layouts util. An example of usage follows below:

    prob.model.add_subsystem('FBWF', 
                         FixedBottomWindFarm(layout_coordinates = np.array([x_coordinates,
                                                                            y_coordinates]),
                                             sim_res = Bastankhah_PorteAgel_2014(site, 
                                                                                 wind_turbines, 
                                                                                 k=0.0324555)),

    """

    def initialize(self):
        self.options.declare("layout_coordinates", 
                             types=np.ndarray, 
                             desc = "Wind farm layout coordinates")
        self.options.declare("sim_res", 
                             desc="xarray from PyWake") # change here to more general

        self.options.declare("n_turbines", 
                             types = int,
                             desc="number of turbines") # change here to more general
        
        self.options.declare('boundary', types=np.ndarray)
        self.options.declare('spacing_diameter', default=6*222, types=(float, int)) # upgrade here for the spacing
        self.options.declare("layout_coordinates", types=np.ndarray)
        self.options.declare("lon_grid_fine", types=np.ndarray)
        self.options.declare("lat_grid_fine",  types=np.ndarray)
        self.options.declare("interpolated_elevation", types=np.ndarray)
        self.options.declare("aep_init", types=xr.DataArray)



    def setup(self):
        # Setting layout coordinates as inputs       
        self.add_input('x', np.zeros(len(self.options["layout_coordinates"][0])))  # X-Layout Coordinates
        self.add_input('y', np.zeros(len(self.options["layout_coordinates"][1])))  # Y-Layout Coordinates
               
        # Setting AEP as output
        self.add_output('AEP', val=0.0)

        # n_turbines = len(self.options['layout_coordinates'])
        n_turbines = len(self.options["layout_coordinates"][0])

    # Declare partial sizes explicitly
        self.declare_partials('AEP', 'x', rows=np.zeros(n_turbines, int), cols=np.arange(n_turbines))
        self.declare_partials('AEP', 'y', rows=np.zeros(n_turbines, int), cols=np.arange(n_turbines))


        x_coordinates = self.options["layout_coordinates"][0]
        y_coordinates = self.options["layout_coordinates"][1]
        boundary =  self.options["boundary"]
        lon_grid_fine = self.options["lon_grid_fine"]
        lat_grid_fine = self.options["lat_grid_fine"]
        interpolated_elevation = self.options["interpolated_elevation"]


        # x = inputs['x']
        # y = inputs['y']



        self.iteration = 0
        self.circles = []
        self.turbine_scatter = None  
        self.cableA = None
        self.cableB = None

        self.fig, self.ax = plt.subplots()

        # if self.iteration == 0:
        

        # Defines the water depth map
        plt.pcolormesh(lon_grid_fine, 
                    lat_grid_fine, 
                    interpolated_elevation, 
                    cmap='Blues_r', 
                    shading='auto', 
                    vmin=-50, 
                    vmax=-20)

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
        self.ax.set_xlabel('X [m]')
        self.ax.set_ylabel('Y [m]')
        # self.ax.set_xlim(360000, 390000)
        # self.ax.set_ylim(4.53E6, 4.56E6)

        self.ax.set_xlim(300000, 350000)
        self.ax.set_ylim(4.54E6, 4.58E6)
        print('done')

        

    def compute(self, inputs, outputs):
        print('Entered compute fixedbottomcomponent')
        outputs['AEP'] = -self.options["sim_res"](inputs['x'], inputs['y']).aep().sum()


        x = inputs['x']
        y = inputs['y']
        boundary =  self.options.declare('boundary')
        # aep_init = -self.options["aep_init"]

        # aep = inputs['AEP'].item()
        aep = outputs['AEP'].item()

        spacing_radius = self.options['spacing_diameter'] / 2






        if self.turbine_scatter is not None:
            self.turbine_scatter.remove()

        if self.cableA is not None:
            for line in self.cableA:
                line.remove()
            self.cableA = None

        if self.cableB is not None:
            for line in self.cableB:
                line.remove()
            self.cableB = None

        # Remove old circles
        for circ in self.circles:
            circ.remove()
        self.circles.clear()

        self.turbine_scatter = self.ax.scatter(x,
                                               y,
                                               marker = '2', 
                                               c='black', 
                                               label='Current Design')

        # Draw new spacing circles
        for xi, yi in zip(x, y):
            circ = Circle((xi, yi), spacing_radius, edgecolor='gray',
                          linestyle='--', facecolor='none', linewidth=1)
            self.ax.add_patch(circ)
            self.circles.append(circ)

        # Draw electrical layout
        VertexC = g1(x,y,boundary).horns.graph['VertexC']
    
        M = g1(x,y,boundary).horns.graph['M']

        X, Y = np.hstack((VertexC[-1:-1 - M:-1].T, VertexC[:-M].T))
        
        Cables = [(-1, 2, 1000), (-1, 4, 1500)]
        
        cable_length = []

        T = heuristic_wrapper(X, Y,Cables,M,heuristic='CPEW')

        T = np.array([[x[0],x[1],x[2],x[3],x[4],Cables[x[4]][2]*x[2]/1000] for x in T])

        for i in range(len(T)):
            # print('cable in meters',T[i][2])
            cable_length.append(T[i][2])

        cable_length = np.array(cable_length).sum()

        Cables = [(-1, 2, 1000), (-1, 4, 1500)]

        ##########################################
        cab0,cab1,cost = [],[],[]

        for i in range(62): #wrong: change here to n_wt, not 62
            if T[i][4] == 0.0:
                cab0.append(i)
                cost.append(Cables[0][2]*T[i][2])
            else:
                cab1.append(i)
                cost.append(Cables[1][2]*T[i][2])

        ##########################################
        total_cable_cost = np.array(cost).sum()

        WTcoords = np.array([x,y])

        WTcentroid = np.array([WTcoords[0].mean(), WTcoords[1].mean()]) #UPDATE THIS TO MATCH REAL
        
        total_cable_cost =  round(total_cable_cost*0.000001, 3) 
        
        # plt.scatter(WTcentroid[0],WTcentroid[1],label='Substation',c='red')
        self.ax.scatter(WTcentroid[0],WTcentroid[1],label='Substation',c='red')

        # colors = ['b','g','r','c','m','y','k','bg','gr','rc','cm']
        colors = ['y', '#b87333' ]

        b = T

        Cables = np.array(Cables)

        for i in range(Cables.shape[0]):
            index = b[:,4]==i
            if index.any():
                n1xs = X[b[index,0].astype(int)-1]
                n2xs = X[b[index,1].astype(int)-1]
                n1ys = Y[b[index,0].astype(int)-1]
                n2ys = Y[b[index,1].astype(int)-1]
                xs = np.vstack([n1xs,n2xs])
                ys = np.vstack([n1ys,n2ys])

                if i == 0:
                    self.cableA = self.ax.plot(xs,ys,'{}'.format(colors[i]),linewidth=1.2)
                
                elif i == 1:
                    self.cableB = self.ax.plot(xs,ys,'{}'.format(colors[i]),linewidth=1.2)

        # Update iteration info
        # self.text_box.set_text(
        #     f"Iteration: {self.iteration}\nAEP Improvement: {((-aep / aep_init) - 1) * 100:.3f} %"
        # )
        # self.text_box.set_text(
        #     f"Iteration: {self.iteration}\nAEP Improvement: {-aep} %"
        # )

        # plt.show()

        plt.draw()
        plt.pause(0.001) 
        # self.ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, fontsize=10)
        # Rebuild legend without duplicates
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))  # removes duplicates based on label
        self.ax.legend(by_label.values(), by_label.keys(),
                    loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, fontsize=10)


        # self.plot_electrical_layout = plot_electrical_cables1(x,y,iter=1)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        self.iteration += 1





    def compute_partials(self, inputs, partials):        
        sim_res = self.options["sim_res"]
        x,y =inputs['x'], inputs['y']

        # Compute exact gradients (PyWake)
        daep = sim_res.aep_gradients(
            gradient_method=autograd,
            wrt_arg=['x', 'y'],
            x = x,
            y = y
        )

        daep_x = daep[0, :]
        daep_y = daep[1, :]

        # Fill OpenMDAO Jacobian
        partials['AEP', 'x'] = -daep_x  # shape (n_turbines,)
        partials['AEP', 'y'] = -daep_y  # shape (n_turbines,)

class OffshoreSystemPlot(om.ExplicitComponent):

    """
    Plot component for an offshore system

    Parameters:
    ----------------------------------------------------------------------------------
    layout_coordinates (float):         x and y wind turbine coordinates 
    sim_res (xarray, float):            Instance of wind farm model in xarray format
    AEP (xarray, float):                Annual Energy Production in xarray format (PyWake)
    ----------------------------------------------------------------------------------

    Usage:
    ----------------------------------------------------------------------------------
    User gives the following arguments when setting the OpenMDAO component:

    """

    def initialize(self):
        self.options.declare('boundary', types=np.ndarray)
        self.options.declare('spacing_diameter', default=6*222, types=(float, int)) # upgrade here for the spacing
        self.options.declare("layout_coordinates", types=np.ndarray)
        self.options.declare("lon_grid_fine", types=np.ndarray)
        self.options.declare("lat_grid_fine",  types=np.ndarray)
        self.options.declare("interpolated_elevation", types=np.ndarray)
        self.options.declare("aep_init", types=xr.DataArray)

    def setup(self):
        x_coordinates = self.options["layout_coordinates"][0]
        y_coordinates = self.options["layout_coordinates"][1]
        boundary =  self.options["boundary"]
        lon_grid_fine = self.options["lon_grid_fine"]
        lat_grid_fine = self.options["lat_grid_fine"]
        interpolated_elevation = self.options["interpolated_elevation"]

        n = len(x_coordinates)  # global or pass via options
        self.add_input('x', np.zeros(n))
        self.add_input('y', np.zeros(n))
        self.add_input('AEP', val=0.0)
        self.add_output('Dummy', val=0)
        self.iteration = 0
        self.circles = []
        self.turbine_scatter = None  
        self.cableA = None
        self.cableB = None


        # # Beginning of the plot definition
        self.fig, self.ax = plt.subplots()
        # plt.close(self.fig)
        
        # # Defines the water depth map
        # plt.pcolormesh(lon_grid_fine, 
        #             lat_grid_fine, 
        #             interpolated_elevation, 
        #             cmap='Blues_r', 
        #             shading='auto', 
        #             vmin=-50, 
        #             vmax=-20)

        # plt.colorbar(label="Water Depth (m)")
        # plt.plot(boundary[:, 0], 
        #          boundary[:, 1], 
        #          label='Boundary', 
        #          c='black', 
        #          linestyle = '--')
        # plt.tight_layout()
        # # plt.ion()
        # self.ax.scatter(x_coordinates,
        #                 y_coordinates, 
        #                 c='orange', 
        #                 marker = '.', 
        #                 s=8, 
        #                 label='Initial Layout')
        # self.text_box = self.ax.text(0.01, 
        #                              0.99, 
        #                              '', 
        #                              transform=self.ax.transAxes, 
        #                              verticalalignment='top', 
        #                              fontsize=10, 
        #                              bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        # self.ax.set_xlabel('X [m]')
        # self.ax.set_ylabel('Y [m]')
        # # self.ax.set_xlim(360000, 390000)
        # # self.ax.set_ylim(4.53E6, 4.56E6)
 
        # self.ax.set_xlim(300000, 350000)
        # self.ax.set_ylim(4.54E6, 4.58E6)
        print('done')


    def compute(self, inputs, outputs):
        x = inputs['x']
        y = inputs['y']
        boundary =  self.options.declare('boundary')
        aep_init = -self.options["aep_init"]

        aep = inputs['AEP'].item()

        spacing_radius = self.options['spacing_diameter'] / 2

        # Remove previous turbine positions (except for the initial layout)
        if self.turbine_scatter is not None:
            self.turbine_scatter.remove()

        if self.cableA is not None:
            for line in self.cableA:
                line.remove()
            self.cableA = None

        if self.cableB is not None:
            for line in self.cableB:
                line.remove()
            self.cableB = None

        # Remove old circles
        for circ in self.circles:
            circ.remove()
        self.circles.clear()

        self.turbine_scatter = self.ax.scatter(x,
                                               y,
                                               marker = '2', 
                                               c='black', 
                                               label='Current Design')

        # Draw new spacing circles
        for xi, yi in zip(x, y):
            circ = Circle((xi, yi), spacing_radius, edgecolor='gray',
                          linestyle='--', facecolor='none', linewidth=1)
            self.ax.add_patch(circ)
            self.circles.append(circ)

        # Draw electrical layout
        VertexC = g1(x,y,boundary).horns.graph['VertexC']
    
        M = g1(x,y,boundary).horns.graph['M']

        X, Y = np.hstack((VertexC[-1:-1 - M:-1].T, VertexC[:-M].T))
        
        Cables = [(-1, 2, 1000), (-1, 4, 1500)]
        
        cable_length = []

        T = heuristic_wrapper(X, Y,Cables,M,heuristic='CPEW')

        T = np.array([[x[0],x[1],x[2],x[3],x[4],Cables[x[4]][2]*x[2]/1000] for x in T])

        for i in range(len(T)):
            # print('cable in meters',T[i][2])
            cable_length.append(T[i][2])

        cable_length = np.array(cable_length).sum()

        Cables = [(-1, 2, 1000), (-1, 4, 1500)]

        ##########################################
        cab0,cab1,cost = [],[],[]

        for i in range(62): #wrong: change here to n_wt, not 62
            if T[i][4] == 0.0:
                cab0.append(i)
                cost.append(Cables[0][2]*T[i][2])
            else:
                cab1.append(i)
                cost.append(Cables[1][2]*T[i][2])

        ##########################################
        total_cable_cost = np.array(cost).sum()

        WTcoords = np.array([x,y])

        WTcentroid = np.array([WTcoords[0].mean(), WTcoords[1].mean()]) #UPDATE THIS TO MATCH REAL
        
        total_cable_cost =  round(total_cable_cost*0.000001, 3) 
        
        # plt.scatter(WTcentroid[0],WTcentroid[1],label='Substation',c='red')
        self.ax.scatter(WTcentroid[0],WTcentroid[1],label='Substation',c='red')

        # colors = ['b','g','r','c','m','y','k','bg','gr','rc','cm']
        colors = ['y', '#b87333' ]

        b = T

        Cables = np.array(Cables)

        for i in range(Cables.shape[0]):
            index = b[:,4]==i
            if index.any():
                n1xs = X[b[index,0].astype(int)-1]
                n2xs = X[b[index,1].astype(int)-1]
                n1ys = Y[b[index,0].astype(int)-1]
                n2ys = Y[b[index,1].astype(int)-1]
                xs = np.vstack([n1xs,n2xs])
                ys = np.vstack([n1ys,n2ys])

                if i == 0:
                    self.cableA = self.ax.plot(xs,ys,'{}'.format(colors[i]),linewidth=1.2)
                
                elif i == 1:
                    self.cableB = self.ax.plot(xs,ys,'{}'.format(colors[i]),linewidth=1.2)

        # Update iteration info
        self.text_box.set_text(
            f"Iteration: {self.iteration}\nAEP Improvement: {((-aep / aep_init) - 1) * 100:.3f} %"
        )
        # plt.show()

        plt.draw()
        plt.pause(0.001) 
        # self.ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, fontsize=10)
        # Rebuild legend without duplicates
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))  # removes duplicates based on label
        self.ax.legend(by_label.values(), by_label.keys(),
                    loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, fontsize=10)


        # self.plot_electrical_layout = plot_electrical_cables1(x,y,iter=1)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        self.iteration += 1

        outputs['Dummy'] = 0