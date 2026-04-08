import pickle
import matplotlib.pyplot as plt

def main():
    open_file = open("Results.pkl", "rb")
    results = pickle.load(open_file)
    open_file.close()

    plt.figure()
    plt.errorbar(results["Key"], results["ANN_Mean"], yerr = results["ANN_StDev"], color='black', capsize = 8, label="ResNet")
    plt.errorbar(results["Key"], results["TRI_Mean"], yerr = results["TRI_StDev"], color='gray', capsize = 8, label = "Trilateration", alpha = 0.5)
    plt.legend()
    plt.savefig("Trilateration_vs_ResNet.png")
    plt.show()

if __name__ == "__main__":
    main()