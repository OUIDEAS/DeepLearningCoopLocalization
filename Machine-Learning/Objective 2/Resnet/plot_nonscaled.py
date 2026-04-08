import matplotlib.pyplot as plt
import pickle

for i in range(7):
    n_anc = i+4
    file = open("NonScaled/"+str(n_anc)+"results.pkl", 'rb')
    results = pickle.load(file)
    file.close()
    print("================================================================")
    print(str(n_anc)+" Anchors")
    print("Training:   ", results['train'][len(results['train'])-1])
    print("Validation: ", results['validate'][len(results['validate'])-1])
    plt.figure()
    plt.plot(results['train'])
    plt.plot(results['validate'])
    plt.xlabel('Epoch [-]')
    plt.ylabel('MAE [m]')
    plt.title(str(n_anc)+" Anchors")

plt.show()
