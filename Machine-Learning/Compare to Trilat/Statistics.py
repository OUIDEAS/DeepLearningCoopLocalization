import pandas as pd
from scipy import stats
import os
os.system('clear')
import itertools
from os.path import isfile, join

class DataLoader():
    def __init__(self, filelist, size):
        first_file=True
        self.file_sizes = {}
        if isinstance(filelist, list):
            for file in filelist:
                file1 = pd.read_csv(file)
                EKF = file1.iloc[0:size, 0].values
                ANN = file1.iloc[0:size, 1].values
                TRI = file1.iloc[0:size, 2].values
                DOP = file1.iloc[0:size, 3].values
                self.file_sizes[file] = len(ANN)

                if first_file:
                    self.ANN = ANN
                    self.EKF = EKF
                    self.TRI = TRI
                    self.DOP = DOP
                    first_file = False

                else:
                    self.ANN = list(itertools.chain(self.ANN,ANN))
                    self.EKF = list(itertools.chain(self.EKF,EKF))
                    self.TRI = list(itertools.chain(self.TRI,TRI))
                    self.DOP = list(itertools.chain(self.DOP,DOP))

                ANN, EKF, TRI, DOP = None, None, None, None

        elif isinstance(filelist, str):
            file1 = pd.read_csv(filelist)
            EKF = file1.iloc[0:size, 0].values
            ANN = file1.iloc[0:size, 1].values
            TRI = file1.iloc[0:size, 2].values
            DOP = file1.iloc[0:size, 3].values
            self.ANN = ANN
            self.EKF = EKF
            self.TRI = TRI
            self.DOP = DOP
            self.file_sizes[filelist] = len(ANN)
        else:
            raise TypeError("Expected list of files or singular file name in the form of a string.")

    def __len__(self):
        return len(self.ANN)




def avg(m):
    return sum(m)/len(m)

def stdev(m):
    return sum((i - avg(m)) ** 2 for i in m) / len(m)

# IF P_VAL > ALPHA: FAIL TO REJECT NULL HYPOTHESIS,
def compare_2_groups(arr_1, first_sample, arr_2, second_sample, alpha, equal_variance = False, alt='less'):
    stat, p = stats.ttest_ind(arr_1, arr_2,equal_var = equal_variance, alternative=alt)
    print('Statistics=%.3f, p=%.3f'%(stat,p))
    if p > alpha:
        print('Same Distribution (fail to reject H_0)')
    else:
        print('Different Distributions (reject (H0))')
        print('Alternative Hypothesis:  ')
        if alt == 'less':
            print('\nthe mean of the distribution underlying the first sample ('+first_sample+') is less than the mean of the distribution underlying the second sample ('+second_sample+').\n')
        elif alt == 'greater':
            print('\nthe mean of the distribution underlying the first sample ('+first_sample+') is greater than the mean of the distribution underlying the second sample ('+second_sample+').\n')
        elif alt == 'two-sided':
            print('\nthe means of the distributions underlying the samples are unequal.\n')
mypath = 'MC_FlightLogs/'
FILES = [mypath+f for f in os.listdir(mypath) if isfile(join(mypath, f))]

results = DataLoader(FILES,10000000000)
tests = ['less', 'two-sided', 'greater']
for t in tests:
    print('---------------------------------------------------------------------')
    if t == 'two-sided':
        print('Current test: two-sided')
    else:
        print('Current test: Neural Network is ', t, ' than Trilateration.')
    compare_2_groups(results.ANN, 'ANN', results.TRI, 'OLS', alpha=0.05, equal_variance=False, alt = t)
    print('---------------------------------------------------------------------\n')

for t in tests:
    print('---------------------------------------------------------------------')
    if t == 'two-sided':
        print('Current test: two-sided')
    else:
        print('Current test: Neural Network is ', t, ' than Kalman Filter OLS.')
    compare_2_groups(results.ANN, 'ANN', results.EKF, 'OLS', alpha=0.05, equal_variance=False, alt = t)
    print('---------------------------------------------------------------------\n')
