import pickle
import matplotlib.pyplot as plt

open_file = open("training_results.pkl", "rb")

t = pickle.load(open_file)

open_file.close()

open_file = open("validation_results.pkl", "rb")

v = pickle.load(open_file)

open_file.close()

plt.figure(2, figsize=[5,5])
plt.plot(v, label='Validation Accuracy')
plt.plot(t, label = 'Training Accuracy')
plt.xlabel('Epoch [-]')
plt.ylabel('Average MAE [m]')
plt.legend()
plt.show()
