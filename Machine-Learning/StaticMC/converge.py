import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import scipy
import pyprog
import itertools
import os
from os.path import isfile, join

class DataLoader():
    def __init__(self, filelist, size):
        first_file=True
        self.file_sizes = {}
        if isinstance(filelist, list):
            for file in filelist:
                file1 = pd.read_csv(file)
                ANN = file1.iloc[0:size, 0].values
                TRI = file1.iloc[0:size, 1].values
                self.file_sizes[file] = len(ANN)

                if first_file:
                    self.ANN = ANN
                    self.TRI = TRI
                    first_file = False

                else:
                    self.ANN = list(itertools.chain(self.ANN,ANN))
                    self.TRI = list(itertools.chain(self.TRI,TRI))

                ANN, TRI, DOP = None, None, None

        elif isinstance(filelist, str):
            file1 = pd.read_csv(filelist)
            ANN = file1.iloc[0:size, 0].values
            TRI = file1.iloc[0:size, 1].values
            self.ANN = ANN
            self.TRI = TRI
            self.file_sizes[filelist] = len(ANN)
        else:
            raise TypeError("Expected list of files or singular file name in the form of a string.")



class FileSize():
    def __init__(self, file1):
        size = 100000000000
        self.name = file1
        file1 = pd.read_csv(file1)
        ANN = file1.iloc[0:size, 0].values
        self.size = len(ANN)
        ANN = None

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

def smooth(vals):
    unfiltered = pd.DataFrame(
        {'unfiltered': vals})
    return unfiltered.ewm(com=100).mean()

def convergence_test(m,title=None, xlabel = None, ylabel=None):
    conv = []
    for i in range(len(m)-2):
        conv.append(m[i+1]/m[i])

    plt.figure()
    plt.plot(conv)
    if title is not None:
        plt.title(title)

    if xlabel is not None:
        plt.xlabel(xlabel)

    if ylabel is not None:
        plt.ylabel(ylabel)

def RMSD(m):
    m_bar = average(m)
    n = len(m)
    sd = 0

    for i in m:
        sd+= (i - m_bar)**2/(n-1)

    return math.sqrt(sd)

def convergence(s, m):
    sw = 5
    rmsd_list = []
    x = []
    for i in range(len(m)-sw):
        leading_edge = int(i+sw)
        trailing_edge = int(i)
        x.append(average(s[trailing_edge:leading_edge]))
        rmsd_list.append(RMSD(m[trailing_edge:leading_edge]))

    return x, rmsd_list

def test_convergence(mypath):
    
    FILES = [mypath+f for f in os.listdir(mypath) if isfile(join(mypath, f))]
    
    ann_avg, ann_stdev, size_list = [], [], []
    tri_avg, tri_stdev = [], []
    count = 0
    results = DataLoader(FILES, 100000000)
    sizes = [1]
    while sizes[len(sizes)-1] < (sum(results.file_sizes[file] for file in FILES)):
        sizes.append(int(2**len(sizes)))
    conv_num = len(sizes)
    prog = pyprog.ProgressBar("-> ", " OK!", conv_num)
    prog.update()
    for j in sizes:
        a = []
        t = []
        results = None
        results = DataLoader(FILES, int(j))
        count += 1
        prog.set_stat(count)
        
        for (a1, t1) in zip(results.ANN, results.TRI):
            a.append(a1)
            t.append(t1)

        size_list.append(j)
        ann_avg.append(sum(a)/len(a))
        ann_stdev.append(stdev(a))
        tri_avg.append(sum(t)/len(t))
        tri_stdev.append(stdev(t))

        prog.update()

    ax, ann_dev = convergence(sizes, ann_stdev)
    bx, ann_mean = convergence(sizes, ann_avg)

    cx, tri_dev = convergence(sizes, tri_stdev)
    dx, tri_mean = convergence(sizes, tri_avg)

    if ann_dev[len(ann_dev)-1] < 0.1 and ann_mean[len(ann_mean)-1] < 0.1 and tri_dev[len(tri_dev)-1] < 0.1 and tri_mean[len(tri_mean)-1] < 0.1:
        converged = True
    else:
        converged = False
    
    return converged, ann_mean[len(ann_mean)-1], ann_dev[len(ann_dev)-1], tri_mean[len(tri_mean)-1], tri_dev[len(tri_dev)-1]
