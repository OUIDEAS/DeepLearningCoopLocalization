import math
import pandas as pd
import os
import matplotlib.pyplot as plt
from os.path import isfile, join
import pprint
import itertools
import numpy as np
import pyprog

class FolderLoader():
    def __init__(self, filelist, size):
        first_file=True
        self.file_sizes = {}
        if isinstance(filelist, list):
            for file in filelist:
                file1 = pd.read_csv(file)
                ANN = file1.iloc[0:size, 0].values
                TRI = file1.iloc[0:size, 1].values
                DOP = file1.iloc[0:size, 2].values
                self.file_sizes[file] = len(ANN)

                if first_file:
                    self.ANN = ANN
                    self.TRI = TRI
                    self.DOP = DOP
                    first_file = False

                else:
                    self.ANN = list(itertools.chain(self.ANN,ANN))
                    self.TRI = list(itertools.chain(self.TRI,TRI))
                    self.DOP = list(itertools.chain(self.DOP,DOP))

                ANN, TRI, POW = None, None, None

        elif isinstance(filelist, str):
            file1 = pd.read_csv(filelist)
            ANN = file1.iloc[0:size, 0].values
            TRI = file1.iloc[0:size, 1].values
            POW = file1.iloc[0:size, 2].values
            self.ANN = ANN
            self.TRI = TRI
            self.DOP = DOP
            self.file_sizes[filelist] = len(ANN)
        else:
            raise TypeError("Expected list of files or singular file name in the form of a string.")

def get_variance(arr):
    mean = sum(arr) / len(arr)
    summ = 0
    for x in arr:
        summ+=(x-mean)**2
    return mean, summ/len(arr)

def average(m):
    return sum(m)/len(m)

def stdev(m):
    _, var = get_variance(m)
    return math.sqrt(var)

def main():
    results = {
        "ANN":{
            "Average":[],
            "StDeviation":[]
        },
        "TRI":{
            "Average":[],
            "StDeviation":[]
        }
    }

    i=0
    mypath = 'MC_FlightLogs/'
    FILES = [mypath+f for f in os.listdir(mypath) if isfile(join(mypath, f))]
    data = FolderLoader(FILES, 10000000000)
    results["ANN"]["Average"].append(average(data.ANN))
    results["ANN"]["StDeviation"].append(stdev(data.ANN))
    results["TRI"]["Average"].append(average(data.TRI))
    results["TRI"]["StDeviation"].append(stdev(data.TRI))
    
    tups = []
    for (a,b,c) in zip(data.ANN, data.TRI, data.DOP):
        tups.append((c, a, b))
    
    tups.sort()
    ann, tri, dop = [], [], []
    for t in tups:
        dop.append(t[0])
        ann.append(t[1])
        tri.append(t[2])
    tavg, aavg = [], []
    tstdev, astdev = [], []

    bins = 100
    bin_size =(20 - tups[0][0])/bins
    dbin = []
    conv_analysis = [[i for i in range(bins)]]
    for i in range(bins):
        min = i * bin_size
        max = (i+1) * bin_size
        a, trilateration = [],[]
        c = 0
        for t in tups:
            if t[0] < max and t[0] > min:
                c+=1
                trilateration.append(t[1])
                a.append(t[2])
        if c > 0:
            dbin.append((min+max)/2)
            tavg.append(sum(trilateration)/len(trilateration))
            aavg.append(sum(a)/len(a))
            tstdev.append(stdev(trilateration))
            astdev.append(stdev(a))


    plt.figure()
    plt.plot(dbin, aavg, color='black', label='ResNet')
    plt.plot(dbin, tavg, color='gray', label='Trilateration')
    plt.xlabel('PDOP [-]')
    plt.ylabel('Mean Absolute Error [m]')
    plt.legend()
    plt.show()

    data = None

    pprint.pprint(results)

    
if __name__ == "__main__":
    main()
