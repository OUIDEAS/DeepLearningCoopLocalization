import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
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
                ER = file1.iloc[0:size, 0].values
                DOP = file1.iloc[0:size, 1].values
                self.file_sizes[file] = len(ER)

                if first_file:
                    self.ER = ER
                    self.DOP = DOP
                    first_file = False

                else:
                    self.ER = list(itertools.chain(self.ER,ER))
                    self.DOP = list(itertools.chain(self.DOP,DOP))

                ER, DOP = None, None

        elif isinstance(filelist, str):
            file1 = pd.read_csv(filelist)
            self.ER = file1.iloc[0:size, 0].values
            self.DOP = file1.iloc[0:size, 1].values
            self.file_sizes[filelist] = len(self.ER)
        else:
            raise TypeError("Expected list of files or singular file name in the form of a string.")



class FileSize():
    def __init__(self, file1):
        size = 100000000000
        self.name = file1
        file1 = pd.read_csv(file1)
        EKF = file1.iloc[0:size, 0].values
        self.size = len(EKF)
        EKF = None

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
mypath = 'MC_FlightLogs/'
FILES = [mypath+f for f in os.listdir(mypath) if isfile(join(mypath, f))]
# FILES = ["1000_MonteCarlo_Results_MAE.csv", "25000_MonteCarlo_Results_MAE.csv","25001_MonteCarlo_Results_MAE.csv", "25002_MonteCarlo_Results_MAE.csv", "25003_MonteCarlo_Results_MAE.csv"]

# Test for convergence in this data set
# 1. Load Data
# 2. Check average Errors for each method in the bins defined for the plots
# 3. Start with a fraction of each sample
# 4. Slowly increase the size you are sampling
# 5. Plot the average error for each bin vs sample size
# 6. Ensure that all lines converge to some steady state value

results = DataLoader('ANN_KF_Combine.csv', 10000000000)

tri = []
dop = []
tup = []

for (a, d) in zip(results.ER, results.DOP):
    tup.append((d, a))
    tri.append(a)
    dop.append(d)


dop_sort = []
tri_sort = []

tmean, tvar = get_variance(tri)

mc_converge_tups = tup
tup.sort()

for t in tup:
    dop_sort.append(t[0])
    tri_sort.append(t[1])

bins = 100
bin_size =(20 - tup[0][0])/bins

tavg = []
dbin = []
conv_analysis = [[i for i in range(bins)]]
for i in range(bins):
    min = i * bin_size
    max = (i+1) * bin_size
    trilateration, c = 0, 0
    for t in tup:
        # if t[0] < 1:
        if t[0] < max and t[0] > min:
            c+=1
            trilateration += t[1]
            
    if c > 0:
        dbin.append((min+max)/2)
        tavg.append(trilateration/c)

plt.figure()
plt.plot(dbin, tavg, linestyle = ':', color='black', label="OLS")
plt.xlabel("PDOP [-]")
plt.xlim([0,20])
plt.ylabel("Mean Absolute Error [m]")
plt.legend()


counta, bins_counta = np.histogram(tri, bins=1000)
pdf = counta / sum(counta)
cdftrilat = np.cumsum(pdf)


print("Trilateration and ANN: ")
for i in range(len(cdftrilat)):
    if cdftrilat[i] >= 0.899 and cdftrilat[i] <= 0.901:
        print(cdftrilat[i],": ", bins_counta[i+1])




plt.figure()
plt.plot(bins_counta[1:], cdftrilat, linestyle = ':', color='black', label="OLS")
plt.xlabel("Mean Absolute Error [m]")
plt.ylabel("CDF [%]")
plt.ylim([0, 1])
plt.xlim([0, 10])
plt.legend()

plt.show()

# Test for convergence in this data set
# 1. Load Data
# 2. Check average Errors for each method in the bins defined for the plots
# 3. Start with a fraction of each sample
# 4. Slowly increase the size you are sampling
# 5. Plot the average error for each bin vs sample size
# 6. Ensure that all lines converge to some steady state value
ann_avg, ann_stdev, size_list = [], [], []
ekf_avg, ekf_stdev = [], []
tri_avg, tri_stdev = [], []
count = 0

sizes = [1]
while sizes[len(sizes)-1] < (sum(results.file_sizes[file] for file in FILES)):
    # sizes.append(2*sizes[len(sizes)-1])
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
    for (a1, e1, t1) in zip(results.ANN, results.EKF, results.TRI):
        a.append(a1)
        e.append(e1)
        t.append(t1)
        c += 1
    if c > 0:
        size_list.append(j)
        ann_avg.append(sum(a)/len(a))
        ann_stdev.append(stdev(a))
        tri_avg.append(sum(t)/len(t))
        tri_stdev.append(stdev(t))
        ekf_avg.append(sum(e)/len(e))
        ekf_stdev.append(stdev(e))

    prog.update()

# convergence_test(ann_avg)
# convergence_test(ann_stdev)
avg_p3s, stdev_p3s = [], []
avg_m3s, stdev_m3s = [], []
stdev_sizes = []
for i in range(int((len(ann_avg)-1)/5)):
    try:
        center = int(i*len(ann_avg)/5)
        s1 = stdev(ann_avg[int(center-len(ann_avg)/5) if center-len(ann_avg)/5 > 0 else 0:center+int(len(ann_avg)/5)])
        s2 = stdev(ann_stdev[int(center-len(ann_avg)/5) if center-len(ann_avg)/5 > 0 else 0:center+int(len(ann_avg)/5)])
        avg_p3s.append(average(ann_avg) + 3*s1)
        stdev_p3s.append(average(ann_stdev) + 3*s2)
        avg_m3s.append(average(ann_avg) - 3*s1)
        stdev_m3s.append(average(ann_stdev) - 3*s2)
        stdev_sizes.append(size_list[int(center)])
    except:
        pass
x_axis, stdev_convergence = convergence(sizes, ann_stdev)
x2_axis, mean_convergence = convergence(sizes, ann_avg)

xe_axis, stdeve_convergence = convergence(sizes, ekf_stdev)
x2e_axis, meane_convergence = convergence(sizes, ekf_avg)

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
plt.ylabel('Root Mean Standard Deviation [m]')

plt.figure()
plt.title('Kalman Filter OLS Convergence')
plt.plot(xe_axis, stdeve_convergence, color = 'black', label = 'Standard Deviation')
plt.plot(x2e_axis, meane_convergence, color = 'black', linestyle = '-.', label = 'Mean')
plt.plot(x_axis, ylims,linestyle=':', color='gray')
plt.xscale('log')
plt.legend()
plt.xlabel('Sample Sizes [-]')
plt.ylabel('Root Mean Standard Deviation [m]')

plt.figure()
plt.title('Trilateration Convergence')
plt.plot(xt_axis, stdevt_convergence, color = 'black', label = 'Standard Deviation')
plt.plot(x2t_axis, meant_convergence, color = 'black', linestyle = '-.', label = 'Mean')
plt.plot(x_axis, ylims,linestyle=':', color='gray')
plt.xscale('log')
plt.legend()
plt.xlabel('Sample Sizes [-]')
plt.ylabel('Root Mean Standard Deviation [m]')

# plt.figure()
# plt.title('Average')
# plt.plot(size_list, ann_avg, color='black', label = "Mean")
# plt.plot(size_list, ann_stdev, color='black', linestyle='-.', label = "Standard Deviation")
# plt.plot(stdev_sizes[:len(avg_p3s)-2], avg_p3s[:len(avg_p3s)-2], color='black', linestyle=':')
# plt.plot(stdev_sizes[:len(avg_p3s)-2], avg_m3s[:len(avg_p3s)-2], color='black', linestyle=':')
# plt.plot(stdev_sizes[:len(avg_p3s)-2], stdev_p3s[:len(avg_p3s)-2], color='black', linestyle=':')
# plt.plot(stdev_sizes[:len(avg_p3s)-2], stdev_m3s[:len(avg_p3s)-2], color='black', linestyle=':')
# # plt.xlim([0, stdev_sizes[len(avg_p3s)-2]])
# plt.xscale('log')
# plt.xlabel('Sample Size [-]')
# plt.ylabel('Absolute Error [m]')


# plot the sample means and sample standard deviations over time
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
