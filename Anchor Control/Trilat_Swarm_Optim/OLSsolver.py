import numpy as np
from random import randrange
import scipy
import numpy as np
import scipy.optimize as opt

def format_data(data):
    r, x, y, z = [],[],[],[]
    for i in data:
        r.append(float(i.r))
        x.append(float(i.x))
        y.append(float(i.y))
        z.append(float(i.z))

    rho = np.array(r)
    points = [[x[0], y[0], z[0]]]
    pass_loop = True
    for (a, b, c) in zip(x, y, z):
        if pass_loop:
            pass_loop = False
        else:
            points.append([a, b, c])

    return rho, np.array(points)
# Trilateration solver using Ordinary Least Squares
# Inputs:
#   rho:    A single column numpy array or the ranges
#   points: A list of lists of each reference points [x, y, z] coordinates

def OLS_Trilat(data):
    rho, points = format_data(data)
    # Initial guess
    u = np.array([[0],[0],[0]])
    # Ordinary Least Squares Solution

    for i in range(5):
        # Create G matrix
        G = np.array([[float((points[0][0]-u[0])/rho[0]),  float((points[0][1]-u[1])/rho[0]),  float((points[0][2]-u[2])/rho[0]) ]])
        for i in range(len(points)-2):
            row = np.array([  float((points[i+1][0]-u[0])/rho[i+1]),  float((points[i+1][1]-u[1])/rho[i+1]),  float((points[i+1][2]-u[2])/rho[i+1]) ])
            G = np.append(G, [row], axis=0)

        # Create dRho matrix
        rho_hat = []
        for p in range(len(points)-1):
            r = float(np.sqrt((points[p][0]-u[0])**2 + (points[p][1]-u[1])**2 + (points[p][2]-u[2])**2))
            rho_hat.append(r)
            if p == 0:
                drho = np.array([[float(rho_hat[0]-rho[0])]])
            else:
                dr = np.array([float(rho_hat[p]-rho[p])])
                drho = np.append(drho, [dr], axis=0)
        # Linear Algebra to adjust the position estimate
        a = np.matmul(np.transpose(G),G)
        du = np.matmul(np.matmul(np.linalg.inv(a), np.transpose(G)), drho)
        # Update estimate
        u = u + du
        # PDOP = float(np.sqrt(G[0][0]**2+G[1][1]**2+G[2][2]**2))
    return u#, PDOP

def get_rho_points(avail):
    r, x, y, z = [],[],[],[]
    for i in avail:
        r.append(float(i.r))
        x.append(float(i.x))
        y.append(float(i.y))
        z.append(float(i.z))

    rho = np.array(r)
    points = [[x[0], y[0], z[0]]]
    pass_loop = True
    for (a, b, c) in zip(x, y, z):
        if pass_loop:
            pass_loop = False
        else:
            points.append([a, b, c])

    return rho, np.array(points)

def PDOP_Solver(pos, avail):
    rho, points = get_rho_points(avail)
    # Calculated Position
    u = pos
   
    # Create G matrix
    G = np.array([[float((points[0][0]-u[0])/rho[0]),  float((points[0][1]-u[1])/rho[0]),  float((points[0][2]-u[2])/rho[0]) ]])
    for i in range(len(points)-2):
        row = np.array([  float((points[i+1][0]-u[0])/rho[i+1]),  float((points[i+1][1]-u[1])/rho[i+1]),  float((points[i+1][2]-u[2])/rho[i+1]) ])
        G = np.append(G, [row], axis=0)

    a = np.linalg.inv(np.matmul(np.transpose(G),G))
    PDOP = float(np.sqrt(a[0][0]**2+a[1][1]**2+a[2][2]**2))
    return PDOP

def lse(X,LandmarkList):
    mse = 0
    lse = 0
    for Landmark in LandmarkList:
        xL = Landmark['KnownLocation'][0]
        yL = Landmark['KnownLocation'][1]
        zL = Landmark['KnownLocation'][2]
        rL = Landmark['Distance']
        error = np.sqrt((xL-X[0])**2 + (yL-X[1])**2 + (zL-X[2])**2)
        lse += (rL - error)**2
        mse = lse/len(LandmarkList)
    return (lse)


def Powell_Trilat(data):
    LandmarkList = makeLandmarkList(data)
    # Initial guess
    # Ordinary Least Squares Solution

    results = scipy.optimize.minimize(lse,[0, 0, 0], method="Powell", args=(LandmarkList),tol=0.000000001)
    x,y,z = results.x
    u = np.array([[x],[y],[z]])
    return u#, PDOP

def makeLandmarkList(data):
    LandmarkList = []
    for i in data:
        LandmarkList.append({
            "KnownLocation": [i.x, i.y, i.z],
            "Distance": i.r
        })
    return LandmarkList
# def main():
#     num_points = 10
#
#     points = []
#     if num_points < 4:
#         print("Using 3 anchors and 1 agent.")
#         for i in range(4):
#             p = [randrange(-500, 500), randrange(-500, 500), randrange(-500,500)]
#             points.append(p)
#     else:
#         for i in range(num_points):
#             p = [randrange(-500, 500), randrange(-500, 500), randrange(-500,500)]
#             points.append(p)
#
#     d = []
#
#     # Create Ranges Matrix
#     for p in range(len(points)-1):
#         dist = float(np.sqrt((points[p][0]-points[-1][0])**2 + (points[p][1]-points[-1][1])**2 + (points[p][2]-points[-1][2])**2))
#         d.append(dist)
#         if p == 0:
#             rho = np.array([[d[0]]])
#         else:
#             r = np.array([d[p]])
#             rho = np.append(rho, [r], axis = 0)
#
#     pos = OLS_Trilat(rho, points)
#     print(pos)
#     print(points[-1])
#
# if __name__ == "__main__":
#     main()
