import pickle
import matplotlib.pyplot as plt

open_file = open("means.pkl", "rb")
v_means = pickle.load(open_file)
open_file.close()   

open_file = open("stdevs.pkl", "rb")
v_stdevs = pickle.load(open_file)
open_file.close()   

open_file = open("keys.pkl", "rb")
key = pickle.load(open_file)
open_file.close()    

print(v_means)
print(key)
plt.figure()
plt.errorbar(key, v_means, yerr=v_stdevs, color='black', capsize=8)
plt.xlabel('Number of Anchors [-]')
plt.ylabel('Validation Error [m]')
plt.show()