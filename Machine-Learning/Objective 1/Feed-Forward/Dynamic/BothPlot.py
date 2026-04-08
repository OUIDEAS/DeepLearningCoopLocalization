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

open_file = open("tri_means.pkl", "rb")
tri_means = pickle.load(open_file)
open_file.close()   

open_file = open("tri_stdevs.pkl", "rb")
tri_stdevs = pickle.load(open_file)
open_file.close()   

open_file = open("tri_keys.pkl", "rb")
tri_key = pickle.load(open_file)
open_file.close()


plt.figure()
plt.errorbar(tri_key, tri_means, yerr=tri_stdevs, color='gray',linestyle = '-.', label='Trilateration', capsize=8, alpha=0.65)
plt.errorbar(key, v_means, yerr=v_stdevs, color='black', label='ANN', capsize=8)
plt.xlabel('Number of Anchors [-]')
# plt.xlim([4,10])
plt.ylabel('Validation Error [m]')
plt.legend()
plt.show()