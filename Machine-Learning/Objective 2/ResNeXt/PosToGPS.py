import numpy as np
from haversine import haversine, Unit
from haversine import inverse_haversine, Direction
import torch

class LLH:
    def __init__(self, lat, lon, h):
        self.lat = lat
        self.lon = lon
        self.h = h
        
def Simulate_GPS(x, y, z):
    # Used Ohio University's Geodetic Coordinates as a reference with height = 0 
    reference_LLH= (39.324360, -82.101387)
    t1 = inverse_haversine(reference_LLH, y, Direction.NORTH, unit=Unit.METERS)
    t2 = inverse_haversine(t1, x, Direction.EAST, unit=Unit.METERS)
    lat = t2[0] + float(np.random.normal(0, 1e-6,1))
    lon = t2[1] + float(np.random.normal(0, 1e-6,1))
    height = z + float(np.random.normal(0, 0.05,1))
    return lat, lon, height

def N(phi):
    r_eq = 6378137 # Earth's Equatorial Radius [meters]
    r_p = 6356752.3142 # Earth's Polar Radius [meters]

    sol = float(r_eq**2/(np.sqrt(r_eq**2*np.cos(phi*np.pi/180)**2 + r_p**2*np.sin(phi*np.pi/180)**2)))
    
    return sol

# Calculate the EAST-NORTH-UP distance between two geodetic points
# Structure point_A and point_B as:
# point.lat
# point.lon
# point.h
# Finds coordinates FROM point_A TO point_B
def GPS_to_ENU(point_A, point_B):
    r_eq = 6378137 # Earth's Equatorial Radius [meters]
    r_p = 6356752 # Earth's Polar Radius [meters]
    # ECEF calculations
    ecef_x_a = (N(point_A.lat) + point_A.h)*np.cos(point_A.lat*np.pi/180)*np.cos(point_A.lon*np.pi/180)
    ecef_y_a = (N(point_A.lat) + point_A.h)*np.cos(point_A.lat*np.pi/180)*np.sin(point_A.lon*np.pi/180)
    ecef_z_a = (r_p**2)/(r_eq**2)*(N(point_A.lat)+point_A.h)*np.sin(point_A.lat*np.pi/180)
    ecef_x_b = (N(point_B.lat) + point_B.h)*np.cos(point_B.lat*np.pi/180)*np.cos(point_B.lon*np.pi/180)
    ecef_y_b = (N(point_B.lat) + point_B.h)*np.cos(point_B.lat*np.pi/180)*np.sin(point_B.lon*np.pi/180)
    ecef_z_b = (r_p**2)/(r_eq**2)*(N(point_B.lat)+point_B.h)*np.sin(point_B.lat*np.pi/180)
    R = np.matrix([[-np.sin(point_A.lon*np.pi/180), np.cos(point_A.lon*np.pi/180), 0], 
                   [-np.sin(point_A.lat*np.pi/180)*np.cos(point_A.lon*np.pi/180), -np.sin(point_A.lat*np.pi/180)*np.sin(point_A.lon*np.pi/180), np.cos(point_A.lat*np.pi/180)], 
                   [np.cos(point_A.lat*np.pi/180)*np.cos(point_A.lon*np.pi/180), np.cos(point_A.lat*np.pi/180)*np.sin(point_A.lon*np.pi/180), np.sin(point_A.lat*np.pi/180)]])
    dECEF = np.matrix([[ecef_x_b - ecef_x_a],
                       [ecef_y_b - ecef_y_a],
                       [ecef_z_b - ecef_z_a]])
    enu = np.matmul(R, dECEF)
    nav = [float(enu[0]), float(enu[1]), float(enu[2])]

    return float(enu[0]), float(enu[1]), float(enu[2])

