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

                ANN,  TRI, DOP = None, None, None

        elif isinstance(filelist, str):
            file1 = pd.read_csv(filelist)
            ANN = file1.iloc[0:size, 0].values
            TRI = file1.iloc[0:size, 1].values
            DOP = file1.iloc[0:size, 2].values
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
        # x.append(average(s[trailing_edge:leading_edge]))
        x.append(s[leading_edge])
        rmsd_list.append(RMSD(m[trailing_edge:leading_edge]))

    return x, rmsd_list
mypath = '/home/rgeng98/BobGeng-Thesis/Machine-Learning/StaticMC/TestAllAnchors/Static_MC_Results_7/'#'Static_MC_Results/'
FILES = [mypath+f for f in os.listdir(mypath) if isfile(join(mypath, f))]
results = DataLoader(FILES, 10000000000)
tri = []
ann = []
dop = []
tup = []
num = 0
den = 0
for (a,b,c) in zip(results.TRI, results.ANN, results.DOP):
    tup.append((c, a, b))
    den = den + 1
    if a > b:
        num = num+1
    tri.append(a)
    ann.append(b)
    dop.append(c)

print("Percentage of ANN<TRI: ",num/den)


dop_sort = []
ann_sort = []
ekf_sort = []
tri_sort = []

tmean, tvar = get_variance(tri)
amean, avar = get_variance(ann)

mc_converge_tups = tup
tup.sort()

for t in tup:
    dop_sort.append(t[0])
    tri_sort.append(t[1])
    ann_sort.append(t[2])

bins = 50
bin_size =(20 - tup[0][0])/bins

tavg, eavg, aavg = [], [], []
dbin = []
conv_analysis = [[i for i in range(bins)]]
for i in range(bins):
    min = i * bin_size
    max = (i+1) * bin_size
    a, trilateration, c = 0, 0, 0
    for t in tup:
        # if t[0] < 1:
        if t[0] < max and t[0] > min:
            c+=1
            trilateration += t[1]
            a += t[2]
    if c > 0:
        dbin.append((min+max)/2)
        tavg.append(trilateration/c)
        aavg.append(a/c)

plt.figure()
plt.plot(dbin, tavg, linestyle = '-.', color='black', label="Trilateration")
plt.plot(dbin, aavg, color='black', label="ResNet")
plt.xlabel("PDOP [-]")
plt.xlim([0,20])
plt.ylabel("Mean Absolute Error [m]")
plt.legend()


counta, bins_counta = np.histogram(tri, bins=1000)
pdf = counta / sum(counta)
cdftrilat = np.cumsum(pdf)

countc, bins_countc = np.histogram(ann, bins=1000)
pdf = countc / sum(countc)
cdfann = np.cumsum(pdf)

print("Trilateration: ")
for i in range(len(cdftrilat)):
    if cdftrilat[i] >= 0.89 and cdftrilat[i] <= 0.91:
        print(cdftrilat[i],": ", bins_counta[i+1])

print("\nANN: ")
for i in range(len(cdfann)):
    if cdfann[i] >= 0.88 and cdfann[i] <= 0.91:
        print(cdfann[i],": ", bins_countc[i+1])


plt.figure()
plt.plot(bins_counta[1:], cdftrilat, linestyle = '-.', color='black', label="Trilateration")
plt.plot(bins_countc[1:], cdfann, color='black', label="ResNet")
plt.xlabel("Mean Absolute Error [m]")
plt.ylabel("CDF [%]")
plt.ylim([0, 1])
plt.xlim([0, 10])
plt.legend()

tmean, tvar = get_variance(results.TRI)
amean, avar = get_variance(results.ANN)

print("ANN MEAN:     ", amean)
print("ANN STDEV:    ", math.sqrt(avar))

print("TRI MEAN:     ", tmean)
print("TRI STDEV:    ", math.sqrt(tvar))


ann_avg, ann_stdev, size_list = [], [], []
tri_avg, tri_stdev = [], []
count = 0

sizes = [1]
while sizes[len(sizes)-1] < (sum(results.file_sizes[file] for file in FILES)):
    sizes.append(int(2**len(sizes)))

conv_num = len(sizes)
prog = pyprog.ProgressBar("-> ", " OK!", conv_num)
prog.update()
for j in sizes:
    a = []
    t, e = [], []
    results = None
    results = DataLoader(FILES, int(j))
    count += 1
    prog.set_stat(count)
    c=0
    for (a1, t1) in zip(results.ANN, results.TRI):
        a.append(a1)
        t.append(t1)
        c += 1
    if c > 0:
        size_list.append(j)
        ann_avg.append(sum(a)/len(a))
        ann_stdev.append(stdev(a))
        tri_avg.append(sum(t)/len(t))
        tri_stdev.append(stdev(t))

    prog.update()

x_axis, stdev_convergence = convergence(sizes, ann_stdev)
x2_axis, mean_convergence = convergence(sizes, ann_avg)

xt_axis, stdevt_convergence = convergence(sizes, tri_stdev)
x2t_axis, meant_convergence = convergence(sizes, tri_avg)

ylims = [0.1 for i in range(len(x_axis))]

plt.figure()
plt.plot(x_axis, stdev_convergence, color = 'black', label = 'Standard Deviation')
plt.plot(x2_axis, mean_convergence, color = 'black', linestyle = '-.', label = 'Mean')
plt.plot(x_axis, ylims,linestyle=':', color='gray')
plt.xscale('log')
plt.legend()
plt.xlabel('Sample Sizes [-]')
plt.ylabel('Root Mean Square Deviation [m]')

plt.figure()
plt.title('Trilateration Convergence')
plt.plot(xt_axis, stdevt_convergence, color = 'black', label = 'Standard Deviation')
plt.plot(x2t_axis, meant_convergence, color = 'black', linestyle = '-.', label = 'Mean')
plt.plot(x_axis, ylims,linestyle=':', color='gray')
plt.xscale('log')
plt.legend()
plt.xlabel('Sample Sizes [-]')
plt.ylabel('Root Mean Square Deviation [m]')

fig, ax = plt.subplots(2, figsize=(8,6))
ax[0].plot(size_list, ann_avg, linestyle = '-.', color='black')
ax[0].set_ylabel('Mean [m]')
ax[0].set_xscale('log')
ax[1].plot(size_list, ann_stdev, color='black')
ax[1].set_xlabel('Sample Size [-]')
ax[1].set_ylabel('Standard Deviation [m]')
ax[1].set_xscale('log')
# fig.suptitle('Monte Carlo Simulation Convergence')
plt.show()

print()
