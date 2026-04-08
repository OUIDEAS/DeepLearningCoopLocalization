import numpy as np
from haversine import haversine, Unit
from haversine import inverse_haversine, Direction
import torch


def Simulate_GPS(x, y, z):
    origin = (0, 0)
    if x < 0:
        tupx = inverse_haversine(origin, abs(x), Direction.WEST, unit=Unit.METERS)
    else:
        tupx = inverse_haversine(origin, abs(x), Direction.EAST, unit=Unit.METERS)
    if y < 0:
        tupy = inverse_haversine(origin, abs(y), Direction.SOUTH, unit=Unit.METERS)
    else:
        tupy = inverse_haversine(origin, abs(y), Direction.NORTH, unit=Unit.METERS)

    lat = tupy[0] + float(np.random.normal(0, 1e-6,1))
    lon = tupx[1] + float(np.random.normal(0, 1e-6,1))
    h = z + float(np.random.normal(0, 0.05,1))

    return lat, lon, h

def GPS_to_XYZ(lat, lon, x, y):
    if x < 0:
        x = -1 * haversine((0,0), (0, lon), unit=Unit.METERS)
    else:
        x = haversine((0,0), (0, lon), unit=Unit.METERS)

    if y < 0:
        y = -1 * haversine((0,0), (lat, 0), unit=Unit.METERS)
    else:
        y = haversine((0,0), (lat, 0), unit=Unit.METERS)


    return x, y

def simulate_GPS_noise(x, y, z):
    origin = (0, 0)
    tupx = inverse_haversine(origin, x, Direction.EAST, unit=Unit.METERS)
    tupy = inverse_haversine(origin, y, Direction.NORTH, unit=Unit.METERS)

    lat = tupy[0] + float(np.random.normal(0, 1e-7,1))
    lon = tupx[1] + float(np.random.normal(0, 1e-7,1))
    h = z + float(np.random.normal(0, 0.05,1))

    x_new = haversine((0,0), (0, lon), unit=Unit.METERS)

    y_new = haversine((0,0), (lat, 0), unit=Unit.METERS)

    return x_new, y_new, h

def Generate_Targets(lat, lon, h, a1_lat, a1_lon, a1_height):
    p1 = (a1_lat, lon)
    a1 = (a1_lat, a1_lon)
    x_target = haversine.haversine(p1, a1, unit=Unit.METERS)

    p2 = (lat, a1_lon)
    a2 = (a1_lat, a1_lon)
    y_target = haversine.haversine(p2, a2, unit=Unit.METERS)

    z_target = h-a1_height

    return torch.tensor([[x_target, y_target, z_target]])

