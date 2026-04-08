import pickle
import matplotlib.pyplot as plt
import torch
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from NNLib import *
import math
import os
import pyprog
import numpy as np
import scipy

class TrainLoader(Dataset):
    def __init__(self, filename: str, num_anc: int):
        n_in = int(num_anc)*6 + 3*(int(num_anc)-1)
        file = pd.read_csv(filename)
        inputs = file.iloc[0:2500000, 0:n_in].values
        targets = file.iloc[0:2500000, n_in:n_in+3].values
        a1 = file.iloc[0:2500000, n_in+3:n_in+6].values
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.a1 = torch.tensor(a1, dtype=torch.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx], self.a1[idx]

def get_variance(arr):
    mean = sum(arr) / len(arr)
    summ = 0
    for x in arr:
        summ+=(x-mean)**2
    return mean, summ/len(arr)

def stdev(m):
    _, var = get_variance(m)
    return math.sqrt(var)

def OLS_Trilat(points, rho):
    u = Powell_Trilat(points, rho)
    return u
    # Initial guess
    u = np.array([[0],[0],[0]])
    # Ordinary Least Squares Solution
    mag_du = 100
    iters = 0
    # while mag_du > 0.01:
    for i in range(100):
        # Create G matrix
        r = np.sqrt(float(points[0][0]-u[0][0])**2 + float(points[0][1]-u[1][0])**2 + float(points[0][2]-u[2][0])**2)

        G = np.array([[float((points[0][0]-u[0][0])/rho[0]),  float((points[0][1]-u[1][0])/rho[0]),  float((points[0][2]-u[2][0])/rho[0]) ]])
        for i in range(len(points)-2):
            row = np.array([  float((points[i+1][0]-u[0][0])/rho[i+1]),  float((points[i+1][1]-u[1][0])/rho[i+1]),  float((points[i+1][2]-u[2][0])/rho[i+1]) ])
            G = np.append(G, [row], axis=0)

        # Create dRho matrix
        rho_hat = []
        for p in range(len(points)-1):
            r = float(np.sqrt((points[p][0]-u[0][0])**2 + (points[p][1]-u[1][0])**2 + (points[p][2]-u[2][0])**2))
            rho_hat.append(r)
            if p == 0:
                drho = np.array([[float(rho_hat[0]-rho[0])]])
            else:
                dr = np.array([float(rho_hat[p]-rho[p])])
                drho = np.append(drho, [dr], axis=0)
        # Linear Algebra to adjust the position estimate
        a = np.matmul(np.transpose(G),G)
        try:
            du = np.matmul(np.matmul(np.linalg.inv(a), np.transpose(G)), drho)
        except:
            # u = Powell_Trilat(points, rho)
            return None

        # Update estimate
        u = u + du
        iters+=1
        mag_du = np.sqrt(du[0][0]**2 + du[1][0]**2 + du[2][0]**2)

    return u

def lse(X,LandmarkList):
    # X   = [[0],[0],[0]]
    # X   = np.array(X)
    # print(X)
    mse = 0
    lse = 0
    # X[2] = Altitude
    Dimensions = 'Three'
    for Landmark in LandmarkList:
        if Dimensions == 'Three':
            xL = Landmark['KnownLocation'][0]
            yL = Landmark['KnownLocation'][1]
            zL = Landmark['KnownLocation'][2]
            rL = Landmark['Distance']
            error = np.sqrt((xL-X[0])**2 + (yL-X[1])**2 + (zL-X[2])**2)
            lse += (rL - error)**2
            mse = lse/len(LandmarkList)
        else:
            xL = Landmark['KnownLocation'][0]
            yL = Landmark['KnownLocation'][1]
            rL = Landmark['Distance']
            L = np.linalg.norm([xL, yL])
            error = np.sqrt((xL-X[0])**2+(yL-X[1])**2)
            lse += (rL - error)**2
    return (lse)


def Powell_Trilat(data, r):
    LandmarkList = makeLandmarkList(data, r)
    results = scipy.optimize.minimize(lse,[0, 0, 0], method="L-BFGS-B", args=(LandmarkList),tol=0.0000000000001)
    x,y,z = results.x
    u = [[x],[y],[z]]
    return u

def makeLandmarkList(data, ranges):
    LandmarkList = []
    for (i,r) in zip(data, ranges):
        LandmarkList.append({
            "KnownLocation": [i[0], i[1], i[2]],
            "Distance": r
        })
    return LandmarkList

def trilateration(x,n):
    ranges = []
    anchors = [[0,0,0]]
    for i in range(n):
        ranges.append(float(x[0][i*6]))

    for i in range(n-1):
        anchors.append([float(x[0][i+n*6]), float(x[0][i+n*6+n-1]), float(x[0][i+n*7+n-2])])

    pos = OLS_Trilat(np.array(anchors), np.array(ranges))
    if pos is not None:
        p = torch.tensor([[pos[0][0], pos[1][0], pos[2][0]]])
        return p
    else:
        return None



def main():
    v_means = []
    v_stdevs = []
    vnn_means = []
    vnn_stdevs = []
    key = []
    loss = torch.nn.L1Loss()
    for i in range(7):
        os.system('clear')
        n_anc = i+4
        model = torch.load(str(n_anc)+' Anchors/FilterLocalizationNetwork-compressed.pt')
        data = TrainLoader(str(n_anc)+' Anchors/Validation_data.csv', num_anc = n_anc)
        validation = DataLoader(data, batch_size = 1, shuffle=False)
        V = []
        v_nn = []
        iteration = 0
        print("Testing Neural Network on " + str(n_anc)+" Anchors.")
        prog = pyprog.ProgressBar("-> ", " OK!", len(data))
        prog.update()
        with torch.no_grad():
            for inputs, target, _ in validation:
                pos = trilateration(inputs, n_anc)
                if pos is not None:
                    error = loss(target, pos)
                    V.append(error.item())
                    inputs, target = inputs.to(torch.device('cuda')), target.to(torch.device('cuda'))
                    pos = model(inputs)
                    error = loss(target, pos)
                    v_nn.append(error.item())
                iteration += 1
                prog.set_stat(iteration)
                prog.update()

        v_means.append(sum(V)/len(V))
        v_stdevs.append(stdev(V))
        vnn_means.append(sum(v_nn)/len(v_nn))
        vnn_stdevs.append(stdev(v_nn))
        key.append(n_anc)

    open_file = open("tri_means.pkl", "wb")
    pickle.dump(v_means, open_file)
    open_file.close()

    open_file = open("tri_stdevs.pkl", "wb")
    pickle.dump(v_stdevs, open_file)
    open_file.close()

    open_file = open("tri_keys.pkl", "wb")
    pickle.dump(key, open_file)
    open_file.close()


    plt.figure()
    plt.errorbar(key, v_means, yerr=v_stdevs, color='gray', alpha=0.5, capsize=8, label='Trilateration')
    plt.errorbar(key, vnn_means, yerr=vnn_stdevs, color='black', capsize=8, label='ResNet')
    plt.xlabel('Number of Anchors [-]')
    plt.ylabel('Validation Error [m]')
    plt.savefig("TrilatANN.png")
    plt.show()

if __name__ == '__main__':
    main()
