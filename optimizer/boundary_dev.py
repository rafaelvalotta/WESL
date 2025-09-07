from windFarms_windTurbines import *
from py_wake.wind_turbines import WindTurbines
from py_wake.flow_map import XYGrid
import os
import geojson
import pyproj
import random
import matplotlib.pyplot as plt


"""
This Script contains utility functions to read geographical boundary data from a GeoJSON file,
convert latitude and longitude coordinates to UTM coordinates, and extract the boundary coordinates
for further processing or visualization.
"""

def geoJson_coordinates_data(filepath):
    with open(filepath, 'r') as file:
        geojson_data = geojson.load(file)
    features = geojson_data["features"]
    geojson_data_geometry = features[0]
    return geojson_data_geometry["geometry"]["coordinates"]

def convert_LatLong_to_utm(lon, lat):
    # Define the WGS84 CRS
    wgs84 = pyproj.CRS('EPSG:4326')
    
    # Determine the UTM zone and hemisphere
    utm_zone = int((lon + 180) / 6) + 1
    hemisphere = 'N' if lat >= 0 else 'S'
    
    # Construct the appropriate EPSG code for UTM
    utm_epsg_code = f'EPSG:{32600 + utm_zone if hemisphere == "N" else 32700 + utm_zone}'
    utm = pyproj.CRS(utm_epsg_code)
    
    # Create a transformer and convert coordinates
    transformer = pyproj.Transformer.from_crs(wgs84, utm, always_xy=True)
    easting, northing = transformer.transform(lon, lat)
    return (easting, northing)



def get_only_boundary(filepath):
    # Extract lat/long coordinates from the file
    coordinates = geoJson_coordinates_data(filepath)

    # Convert each (lon, lat) pair to UTM (easting, northing)
    utm_coords = [convert_LatLong_to_utm(lon, lat) for lon, lat in coordinates]

    # Optionally, close the boundary by appending the first point at the end
    if utm_coords[0] != utm_coords[-1]:
        utm_coords.append(utm_coords[0])

    # Unzip the list of tuples into two lists for plotting
    eastings, northings = zip(*utm_coords)
    return eastings, northings