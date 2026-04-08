import pandas as pd
from sklearn.preprocessing import StandardScaler
import scipy
import math
import inspect
import os
import matplotlib.pyplot as plt

def get_variance(arr):
    mean = sum(arr) / len(arr)
    summ = 0
    for x in arr:
        summ+=(x-mean)**2
    return summ/len(arr)

def get_er(m):
    avg = mean(m)
    er = []
    for i in m:
        er.append(i-avg)
    return er

class DataLoader():
    def __init__(self, filename):
        file = pd.read_csv(filename)
        ANN = file.iloc[0:1000000, 0].values
        EKF = file.iloc[0:1000000, 1].values
        TRI = file.iloc[0:1000000, 2].values
        self.ANN = list(ANN)
        self.EKF = list(EKF)
        self.TRI = list(TRI)
       
def mean(sample):
    return sum(sample)/len(sample)

def stdev(arr):
    return math.sqrt(get_variance(arr))

def get_tstat(sample1, sample2):
    num = mean(sample1)-mean(sample2)
    den = math.sqrt(get_variance(sample1)/len(sample1) + get_variance(sample2)/len(sample2))
    return num/den

os.system('clear')

results = DataLoader('Results_MAE.csv')
print('\nPerform a student\'s t-test on the Absolute Error of the different methods')
print('of localization to identify if one method is better\n')
print('ANN vs EKF -----------------------------------\n')
print('Null Hypothesis: Equal MAE')
print('Alternative Hypothesis: ANN MAE > EKF MAE\n')
q = 1-0.05
t_crit = scipy.stats.t.ppf(q=1-0.05, df=len(results.ANN)-1)
t_stat, pval = scipy.stats.ttest_ind(results.ANN, results.EKF, equal_var=True, alternative='two-sided', axis=None)
print('Calculated t-statistic:    ',t_stat)
print('P-Value:                   ',pval)
print('T-Critical Value:          ',t_crit)
print('')

# if abs(t_stat) > abs(t_crit):
#     print("Reject the null hypothesis\n")
# else:
#     print("Fail to reject the null hypothesis\n")

if abs(pval) <= abs(q):
    print("Reject the null hypothesis\n")
else:
    print("Fail to reject the null hypothesis\n")

avar = get_variance(results.ANN)
print('ANN Variance:              ', avar)
print('ANN Mean:                  ', mean(results.ANN))

evar = get_variance(results.EKF)
print('EKF Variance:              ', evar)
print('EKF Mean:                  ', mean(results.EKF))
print('\nANN vs Trilateration -------------------------\n')
t_stat, pval = scipy.stats.ttest_ind(results.ANN, results.TRI, equal_var=False, alternative='greater', axis=None)
print('Calculated t-statistic:    ',t_stat)
print('P-Value:                   ',pval)
print('')
if t_stat > t_crit:
    print("Reject the null hypothesis\n")
else:
    print("Fail to reject the null hypothesis\n")
avar = get_variance(results.ANN)
print('ANN Variance:              ', avar)
print('ANN Mean:                  ', mean(results.ANN))

evar = get_variance(results.TRI)
print('Trilateration Variance:    ', evar)
print('Trilateration Mean:        ', mean(results.TRI))
