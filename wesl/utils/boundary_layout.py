import matplotlib.pyplot as plt                                 # Python library to support plots/graphs
import geopandas as gpd                                         # Geo-spatial data library
from shapely.geometry import LineString                         # Manipulation anad analysis of planar geometric objects
import pyproj                                                   # Cartographic projections and coordinates transformation
import os                                                       # Handles UNIX or NT operating system routines    
import numpy as np                                              # Numerical python for enhanced vectorial/arrays operations
import geojson                                                  # Library with formatting for encoding geographic data structures
import sys                                                      # Access to variables and functions used by the Python interpreter
import pickle 
# from offshore_wind_farms.vineyard_wind import x_vineyard, y_vineyard, boundary_vineyard, SG_14222, VineyardWind
from wesl.offshore_wind_farms.all_wind_farms import wind_farms_europe


def plot_farm_layout(farm):
    """

    This function takes a list of wind farms and returns (x,y) layout coordinates. 
    Additionally, the function plots the layout in UTM coordinates.

    Parameters
    ----------
    farm (string): list of wind farms
    wt_x (float) : layout x-coordinates
    wt_y (float) : layout y-coordinates

    """

    # base_dir = os.getcwd()

    farm_name = wind_farms_europe[farm]
    
    filepath = os.getcwd()
    # directory1 = os.path.dirname(os.path.abspath(__file__))

    # filepath = os.path.join(directory,'turbine_layouts')
    directory = os.path.join(filepath,'turbine_layouts')


    # Split the file name and extension
    name, ext = os.path.splitext(farm_name)
    

    # Add '_TBL' before the extension
    farm_name_with_tbl = f"{name}_TBL{ext}"

    # geojson_filepath = os.path.join(directory, 'turbine_layouts', farm_name_with_tbl)
    geojson_filepath = os.path.join(directory, farm_name_with_tbl)

    # position_Lat_long = geoJson_coordinates_data1(geojson_filepath)

    position_Lat_long = get_lat_long(geojson_filepath)

    wt_x1, wt_y1 = [], [] #lat and long

    for i in range(len(position_Lat_long)):
        wt_x1.append(position_Lat_long[i][0]) #lat and long
        wt_y1.append(position_Lat_long[i][1]) #lat and long


    wt_x, wt_y = [], [] #utm

    # Convert boundary coordinates from latitude/longitude to UTM coordinates
    for i in range(len(position_Lat_long)):
        turbine_pos = convert_LatLong_to_utm(position_Lat_long[i][0], position_Lat_long[i][1])

        wt_x.append(turbine_pos[0]) #utm
        wt_y.append(turbine_pos[1]) #utm
    
    return wt_x, wt_y

def convert_LatLong_to_utm(long, lat):
    """

    This function takes longitude and latitude in degrees and convert to UTM coordinates.
    
    Parameters
    ----------
    long (float): longitude of the coordinate (either boundary or turbine)
    lat  (float): latitude of the coordinate (either boundary or turbine)

    """
    # Create WGS84 (lat/lon) projection
    wgs84 = pyproj.CRS('EPSG:4326')
    
    # Determine the UTM zone and whether it's in the northern or southern hemisphere
    utm_zone = int((long + 180) / 6) + 1
    hemisphere = 'N' if lat >= 0 else 'S'
    
    # Create UTM projection
    # EPSG codes for UTM zones range from 32601 to 32660 for northern hemisphere and from 32701 to 32760 for southern hemisphere
    utm_epsg_code = f'EPSG:{32600 + utm_zone if hemisphere == "N" else 32700 + utm_zone}'
    utm = pyproj.CRS(utm_epsg_code)
    
    # Create transformer
    transformer = pyproj.Transformer.from_crs(wgs84, utm, always_xy=True)
    
    # Perform transformation
    easting, northing = transformer.transform(long, lat)
    
    return (easting, northing)

def geoJson_coordinates_data(farm_name, continent):

    """
    This function returns geojson raw data given a list of wind farms and information on continental location.

    Parameters
    ----------
    farm_name (string): list of wind farms 
    continent (string): Europe and USA at the moment (will extend to Asia and South America)
    
    """

    # current_directory = os.path.dirname(sys.argv[0])
    # base_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.getcwd()


    if continent == 'europe':
        europe = 'boundaries_europe/europe'
        filepath = os.path.join(base_dir,europe)
    
    elif continent == 'usa':
        usa = 'wesl/boundaries_usa'
        filepath = os.path.join(base_dir,usa)

    filepath1 = os.path.join(filepath, farm_name)

    with open(filepath1, 'r') as file:
        geojson_data = geojson.load(file)

    return geojson_data


def get_lat_long(filepath):
    """
    This function supports getting latitude and longitude coordinates from a specific geojson file.
    All in all, it is meant to support post-processing of raw geojson data.

    Parameters
    ----------
    filepath(string): path of the geojson file 

    """


    with open(filepath, 'r') as file:
        geojson_data = geojson.load(file)

    features = geojson_data["features"]
    geojson_data_geometry = features[0]
    return geojson_data_geometry["geometry"]["coordinates"]


def plot_bound(farm_names, list_farms):
    """
    This function plots the boundaries of a list of wind farms.

    Parameters
    ----------
    farm_names (string): list of names of wind farms desired for plotting boundaries in UTM coordinates
    """
    D = 220

    turbine_pos1 = []

    fig, ax = plt.subplots()
    j = 0

    
    for i in farm_names:

        coordinates = i["features"][0]["geometry"]["coordinates"]
        turbine_pos1 = []
        for k in range(len(coordinates)):
            turbine_pos = []
            a = convert_LatLong_to_utm(coordinates[k][0], coordinates[k][1])
            turbine_pos.append(a[0])
            turbine_pos.append(a[1])

            turbine_pos1.append(turbine_pos)

        # with open('utm_boundary_vw_oct25th.pkl', 'wb') as f:
        with open(list_farms[0]+'_boundary.pkl', 'wb') as f:
            pickle.dump(turbine_pos1, f)

        line = LineString(turbine_pos1)
        gfd = gpd.GeoDataFrame({"geometry": [line]})
        gfd.plot(ax=ax, color='black', linewidth=2)
    
    for j in range(len(list_farms)):
        coords = plot_farm_layout(list_farms[j])
        bx1 = np.asarray(min(coords[0]))
        by1 = np.asarray(min(coords[1]))
        bx1 = 0
        by1 = 0
        bx=np.asarray(coords[0])-bx1
        by=np.asarray(coords[1])-by1
        ax.scatter(bx, by, marker='.', c='black', s=5)

        # with open('utm_layout_vw_oct25th.pkl', 'wb') as f:
        with open(list_farms[0]+'_layout.pkl', 'wb') as f:
            pickle.dump(coords, f)


    ax.set_xlabel("X-UTM Coordinates", fontsize=12)
    ax.set_ylabel("Y-UTM Coordinates", fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)
    plt.grid(True)
    plt.show(block=True)


def farm_to_get_boundary_and_layout(farm_name):

    geojson_data, farm_names = [],[]

    # for i in wind_farms_europe:
    #     farm_names.append(geoJson_coordinates_data(wind_farms_europe[i], 'usa'))


    # list_wind_farms_europe = []

    # for k in range(len(wind_farms_europe)):
    #     list_wind_farms_europe.append(list(wind_farms_europe.items())[k][0])
    
    for i in farm_name:
        farm_names.append(geoJson_coordinates_data(farm_name[i], 'usa'))


    list_wind_farms_europe = []

    for k in range(len(farm_name)):
        list_wind_farms_europe.append(list(farm_name.items())[k][0])


    return farm_names, list_wind_farms_europe


def main():
    if __name__ == '__main__':
        vars = farm_to_get_boundary_and_layout(wind_farms_europe)
        plot_bound(vars[0], vars[1])


main()


# with open('utm_boundary_vw_oct25th.pkl', 'rb') as f:
#     boundary = np.array(pickle.load(f))

# with open('utm_layout_vw_oct25th.pkl', 'rb') as f:
#     xinit,yinit = np.array(pickle.load(f))


