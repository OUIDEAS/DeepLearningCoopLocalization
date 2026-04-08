import numpy as np
from random import randrange

def format_tensor(data):
    r = []
    for i in range(10):
        r.append(float(data[0][i*6]))
    x, y, z = [],[],[]
    for a in range(9):
        x.append(float(data[0][6*i + 6 + a*18]))
        y.append(float(data[0][6*i + 12 + a*18]))
        z.append(float(data[0][6*i + 18 + a*18]))

    rho = np.array(r)
    points = [[0,0,0]]
    for a in range(9):
        points.append([x[a], y[a], z[a]])

    return rho, np.array(points)
# Trilateration solver using Ordinary Least Squares
# Inputs:
#   rho:    A single column numpy array or the ranges
#   points: A list of lists of each reference points [x, y, z] coordinates

def OLS_Trilat(data):
    rho, points = format_tensor(data)
    # Initial guess
    u = np.array([[0],[0],[0]])
    # Ordinary Least Squares Solution

    for i in range(25):
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
        PDOP = float(np.sqrt(G[0][0]**2+G[1][1]**2+G[2][2]**2))
    return u#, PDOP



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
