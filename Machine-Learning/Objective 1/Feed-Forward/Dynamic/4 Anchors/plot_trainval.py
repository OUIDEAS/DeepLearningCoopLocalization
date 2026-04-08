import pickle
import matplotlib.pyplot as plt

open_file = open("training_results.pkl", "rb")

t = pickle.load(open_file)

open_file.close()

open_file = open("validation_results.pkl", "rb")

v = pickle.load(open_file)

open_file.close()

print("Training:        ", t[len(t)-1])
print("Validation:      ", v[len(v)-1])

plt.figure()
plt.plot(v, color='black', label='Validation Accuracy')
plt.plot(t, color='gray', label = 'Training Accuracy')
plt.xlabel('Epoch [-]')
plt.ylabel('Average MAE [m]')
plt.legend()
plt.show()
