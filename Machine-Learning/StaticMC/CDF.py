import numpy as np
import matplotlib.pyplot as plt
import itertools
import pandas as pd

class FolderLoader():
    def __init__(self, filelist, folder, size):
        first_file=True
        self.file_sizes = {}
        if isinstance(filelist, list):
            for file in filelist:
                file = folder+file
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
        
def CDF(m):
    count, bins_count = np.histogram(m, bins=1000000)
    pdf = count / sum(count)
    cdf = np.cumsum(pdf)
    return bins_count[1:], cdf
            
import os
def main():
    print("Loading Data")
    folder = "TestAllAnchors/Static_MC_Results_7/"
    from os.path import isfile, join
    files = [f for f in os.listdir(folder) if isfile(join(folder, f))]
    data = FolderLoader(files, folder, 100000000)
    ann_x, ann_y = CDF(data.ANN)
    tri_x, tri_y = CDF(data.TRI)
    plt.figure()
    plt.plot(ann_x, ann_y, color = 'black', label = "ResNet")
    plt.plot(tri_x, tri_y, color = 'gray', label="Trilateration")
    plt.xlabel("Mean Absolute Error [m]")
    plt.ylabel("CDF [%]")
    plt.ylim([0, 1])
    plt.xlim([0,10])
    plt.legend()
    plt.show()
    

if __name__ == "__main__":
    main()
