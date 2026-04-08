import pickle
import matplotlib.pyplot as plt

def main():
    # for i in range(7):
    #     open_file = open(str(i+4)+"training_results.pkl", "rb")

    #     t = pickle.load(open_file)

    #     open_file.close()

    #     open_file = open(str(i+4)+"validation_results.pkl", "rb")

    #     v = pickle.load(open_file)

    #     open_file.close()
    #     print("\n============================================")
    #     print(str(i+3)+" Anchors: ")
    #     print("Training:        ", t[len(t)-1])
    #     print("Validation:      ", v[len(v)-1])

    #     plt.figure()
    #     plt.plot(v, color='black', label = 'Validation Accuracy')
    #     plt.plot(t, color='gray',  label = 'Training Accuracy')
    #     plt.xlabel('Epoch [-]')
    #     plt.ylabel('Average MAE [m]')
    #     plt.legend()
    print("\n============================================")
    open_file = open("training_results.pkl", "rb")

    t = pickle.load(open_file)

    open_file.close()

    open_file = open("validation_results.pkl", "rb")

    v = pickle.load(open_file)

    open_file.close()
    print("\n============================================")
    print("Normalized Inputs: ")
    print("Training:        ", t[len(t)-1])
    print("Validation:      ", v[len(v)-1])

    plt.figure()
    plt.plot(v, color='black', label = 'Validation Accuracy')
    plt.plot(t, color='gray',  label = 'Training Accuracy')
    plt.xlabel('Epoch [-]')
    plt.ylabel('Average MAE [m]')
    plt.title('Normalized Inputs')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
