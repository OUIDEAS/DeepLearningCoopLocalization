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
                POW = file1.iloc[0:size, 2].values
                self.file_sizes[file] = len(ANN)

                if first_file:
                    self.ANN = ANN
                    self.TRI = TRI
                    self.POW = POW
                    first_file = False

                else:
                    self.ANN = list(itertools.chain(self.ANN,ANN))
                    self.TRI = list(itertools.chain(self.TRI,TRI))
                    self.POW = list(itertools.chain(self.POW,POW))

                ANN, TRI, POW = None, None, None

        elif isinstance(filelist, str):
            file1 = pd.read_csv(filelist)
            ANN = file1.iloc[0, 0:size].values
            TRI = file1.iloc[1, 0:size].values
            POW = file1.iloc[2, 0:size].values
            self.ANN = ANN
            self.TRI = TRI
            self.POW = POW
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
        "key":[],
        "ANN":{
            "Average":[],
            "StDeviation":[]
        },
        "TRI":{
            "Average":[],
            "StDeviation":[]
        },
        "POW":{
            "Average":[],
            "StDeviation":[]
        }
    }
    i=0
    for i in range(6):
        num_anc = i+4
        mypath = 'Static_MC_Results_'+str(num_anc)+'/'
        FILES = [mypath+f for f in os.listdir(mypath) if isfile(join(mypath, f))]
        data = FolderLoader(FILES, 10000000000)
        results["ANN"]["Average"].append(average(data.ANN))
        results["ANN"]["StDeviation"].append(stdev(data.ANN))
        results["TRI"]["Average"].append(average(data.TRI))
        results["TRI"]["StDeviation"].append(stdev(data.TRI))
        results["POW"]["Average"].append(average(data.POW))
        results["POW"]["StDeviation"].append(stdev(data.POW))
        results["key"].append(num_anc)
        data = None

    pprint.pprint(results)

    plt.figure()
    plt.errorbar(results["key"], results["ANN"]["Average"], yerr=results["ANN"]["StDeviation"], color="black", capsize = 8, label="ResNet")
    plt.errorbar(results["key"], results["TRI"]["Average"], yerr=results["TRI"]["StDeviation"], color="gray", capsize = 8, label="Trilateration", alpha=0.75)
    plt.errorbar(results["key"], results["POW"]["Average"], yerr=results["POW"]["StDeviation"], color="gray", capsize = 8, label="Powell optimizer", alpha=0.5)
    plt.legend()
    plt.xlabel('Number of Anchors [-]')
    plt.ylabel('Error [m]')

    plt.figure()
    plt.errorbar(results["key"], results["ANN"]["Average"], yerr=results["ANN"]["StDeviation"], color="black", capsize = 8, label="ResNet")
    # plt.errorbar(results["key"], results["TRI"]["Average"], yerr=results["TRI"]["StDeviation"], color="gray", capsize = 8, label="Trilateration", alpha=0.75)
    plt.errorbar(results["key"], results["POW"]["Average"], yerr=results["POW"]["StDeviation"], color="gray", capsize = 8, label="Trilateration", alpha=0.75)
    plt.legend()
    plt.xlabel('Number of Anchors [-]')
    plt.ylabel('Error [m]')
    plt.show()

if __name__ == "__main__":
    main()
