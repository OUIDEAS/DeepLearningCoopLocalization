import torch
import torch.nn as nn
import torch.optim as optim
from NNLib import *
import os
import time
from DLoaderRI import *
from torch.utils.data import DataLoader
import random

global learning_rate
learning_rate = 0.001

def AverageError(acc):
    N = 100
    average_error = sum(acc[-N:]) / N
    return average_error


def fitness_func(err, verr, size, layers):
    # Minimize error without massively expanding network
    fitness = 1/(0.1*err+0.9*verr)#/2) #(10000/err) - 0.00001*size*layers
    return fitness


def main():
    start_time = time.ctime()
    os.system('clear')
    n_generations = 75
    epochs = 200
    n_in = 204
    train = TrainLoader('Training_data.csv')
    validate = TrainLoader('Validation_data.csv')
    train_loader = DataLoader(train, batch_size=500, shuffle=True)
    validation_loader = DataLoader(validate, batch_size = 500, shuffle = True)
    # Create 10 nn architectures

    # Number of hidden Layers
    n_HL = torch.randint(low=5, high=10, size=(1, 10))

    # Adjust the tensor containing the number of hidden layers to a 1 dimensional tensor
    n_HL = torch.tensor([n_HL[0][0], n_HL[0][1], n_HL[0][2], n_HL[0][3], n_HL[0][4], n_HL[0][5], n_HL[0][6], n_HL[0][7],
                         n_HL[0][8], n_HL[0][9]])

    # Hidden size
    n_HS = torch.randint(low=500, high=1000, size=(1, 10))

    # Adjust the tensor containing the number of neurons per layer to a 1 dimensional tensor
    n_HS = torch.tensor([n_HS[0][0], n_HS[0][1], n_HS[0][2], n_HS[0][3], n_HS[0][4], n_HS[0][5], n_HS[0][6], n_HS[0][7],
                         n_HS[0][8], n_HS[0][9]])

    # Define neural networks
    # CPU is used as the device for these networks because GPU on this computer does not have enough memory to
    # handle all of the networks.
    nn1 = NeuralNetwork(n_inputs=n_in, hidden_size=int(n_HS[0]), num_layers=int(n_HL[0]))
    GA1 = Localization(nn1, optim.Adam(nn1.parameters(), lr=learning_rate), nn.MSELoss(), nn.MSELoss())

    nn2 = NeuralNetwork(n_inputs=n_in, hidden_size=int(n_HS[1]), num_layers=int(n_HL[1]))
    GA2 = Localization(nn2, optim.Adam(nn2.parameters(), lr=learning_rate), nn.MSELoss(), nn.MSELoss())

    nn3 = NeuralNetwork(n_inputs=n_in, hidden_size=int(n_HS[2]), num_layers=int(n_HL[2]))
    GA3 = Localization(nn3, optim.Adam(nn3.parameters(), lr=learning_rate), nn.MSELoss(), nn.MSELoss())

    nn4 = NeuralNetwork(n_inputs=n_in, hidden_size=int(n_HS[3]), num_layers=int(n_HL[3]))
    GA4 = Localization(nn4, optim.Adam(nn4.parameters(), lr=learning_rate), nn.MSELoss(), nn.MSELoss())

    nn5 = NeuralNetwork(n_inputs=n_in, hidden_size=int(n_HS[4]), num_layers=int(n_HL[4]))
    GA5 = Localization(nn5, optim.Adam(nn5.parameters(), lr=learning_rate), nn.MSELoss(), nn.MSELoss())

    nn6 = NeuralNetwork(n_inputs=n_in, hidden_size=int(n_HS[5]), num_layers=int(n_HL[5]))
    GA6 = Localization(nn6, optim.Adam(nn6.parameters(), lr=learning_rate), nn.MSELoss(), nn.MSELoss())

    nn7 = NeuralNetwork(n_inputs=n_in, hidden_size=int(n_HS[6]), num_layers=int(n_HL[6]))
    GA7 = Localization(nn7, optim.Adam(nn7.parameters(), lr=learning_rate), nn.MSELoss(), nn.MSELoss())

    nn8 = NeuralNetwork(n_inputs=n_in, hidden_size=int(n_HS[7]), num_layers=int(n_HL[7]))
    GA8 = Localization(nn8, optim.Adam(nn8.parameters(), lr=learning_rate), nn.MSELoss(), nn.MSELoss())

    nn9 = NeuralNetwork(n_inputs=n_in, hidden_size=int(n_HS[8]), num_layers=int(n_HL[8]))
    GA9 = Localization(nn9, optim.Adam(nn9.parameters(), lr=learning_rate), nn.MSELoss(), nn.MSELoss())

    nn10 = NeuralNetwork(n_inputs=n_in, hidden_size=int(n_HS[9]), num_layers=int(n_HL[9]))
    GA10 = Localization(nn10, optim.Adam(nn10.parameters(), lr=learning_rate), nn.MSELoss(), nn.MSELoss())

    besthl = []
    besths = []
    for i in range(n_generations):
        print("Start time:  ", start_time)
        print("Generation:  ", i+1)
        e1, e2, e3, e4, e5, e6, e7, e8, e9, e0 = [], [], [], [], [], [], [], [], [], []
        # Train All networks first
        acc1 = []
        acc2 = []
        acc3 = []
        acc4 = []
        acc5 = []
        acc6 = []
        acc7 = []
        acc8 = []
        acc9 = []
        acc0 = []
        for e in range(epochs):
            a = 0
            for data, target, a1 in train_loader:
                pos1, t1 = GA1.train(data, target)
                pos2, t2 = GA2.train(data, target)
                pos3, t3 = GA3.train(data, target)
                pos4, t4 = GA4.train(data, target)
                pos5, t5 = GA5.train(data, target)
                pos6, t6 = GA6.train(data, target)
                pos7, t7 = GA7.train(data, target)
                pos8, t8 = GA8.train(data, target)
                pos9, t9 = GA9.train(data, target)
                pos10, t10 = GA10.train(data, target)
                acc1.append(t1)
                acc2.append(t2)
                acc3.append(t3)
                acc4.append(t4)
                acc5.append(t5)
                acc6.append(t6)
                acc7.append(t7)
                acc8.append(t8)
                acc9.append(t9)
                acc0.append(t10)



        # Compare performance based on validation data
        for data, target, a1 in validation_loader:
            pos1, v1 = GA1.test(data, target)
            pos2, v2 = GA2.test(data, target)
            pos3, v3 = GA3.test(data, target)
            pos4, v4 = GA4.test(data, target)
            pos5, v5 = GA5.test(data, target)
            pos6, v6 = GA6.test(data, target)
            pos7, v7 = GA7.test(data, target)
            pos8, v8 = GA8.test(data, target)
            pos9, v9 = GA9.test(data, target)
            pos10, v10 = GA10.test(data, target)
            e1.append(v1)
            e2.append(v2)
            e3.append(v3)
            e4.append(v4)
            e5.append(v5)
            e6.append(v6)
            e7.append(v7)
            e8.append(v8)
            e9.append(v9)
            e0.append(v10)


        av1 = AverageError(acc1)
        av2 = AverageError(acc2)
        av3 = AverageError(acc3)
        av4 = AverageError(acc4)
        av5 = AverageError(acc5)
        av6 = AverageError(acc6)
        av7 = AverageError(acc7)
        av8 = AverageError(acc8)
        av9 = AverageError(acc9)
        av0 = AverageError(acc0)
        # Average Error for validation data
        ae1 = sum(e1) / len(e1)
        ae2 = sum(e2) / len(e2)
        ae3 = sum(e3) / len(e3)
        ae4 = sum(e4) / len(e4)
        ae5 = sum(e5) / len(e5)
        ae6 = sum(e6) / len(e6)
        ae7 = sum(e7) / len(e7)
        ae8 = sum(e8) / len(e8)
        ae9 = sum(e9) / len(e9)
        ae0 = sum(e0) / len(e0)

        f1 = fitness_func(ae1, av1, n_HS[0], n_HL[0])
        f2 = fitness_func(ae2, av2, n_HS[1], n_HL[1])
        f3 = fitness_func(ae3, av3, n_HS[2], n_HL[2])
        f4 = fitness_func(ae4, av4, n_HS[3], n_HL[3])
        f5 = fitness_func(ae5, av5, n_HS[4], n_HL[4])
        f6 = fitness_func(ae6, av6, n_HS[5], n_HL[5])
        f7 = fitness_func(ae7, av7, n_HS[6], n_HL[6])
        f8 = fitness_func(ae8, av8, n_HS[7], n_HL[7])
        f9 = fitness_func(ae9, av9, n_HS[8], n_HL[8])
        f0 = fitness_func(ae0, av0, n_HS[9], n_HL[9])


        # Put all the final losses along with a corresponding number into a list of tuples
        rank = [(f1, 0),
                (f2, 1),
                (f3, 2),
                (f4, 3),
                (f5, 4),
                (f6, 5),
                (f7, 6),
                (f8, 7),
                (f9, 8),
                (f0, 9)]

        # Sort the list with the highest fitness appearing earlier in the array
        rank.sort(reverse=True)
        GA1, GA2, GA3, GA4, GA5, GA6, GA7, GA8, GA9, GA10, n_HL, n_HS = mutate(rank, n_HL, n_HS, i+1, n_in)
        besthl.append(int(n_HL[0]))
        besths.append(int(n_HS[0]))
        os.system('clear')
        print("Best Architecture:")
        print("Top number of hidden layers of each generation:")
        print(besthl)
        print("Top hidden sizes of each generation:")
        print(besths)
        print("\n")

    os.system('clear')
    end_time = time.ctime()
    print("Start:   ", start_time)
    print("End:     ", end_time)
    print("Best Performing Hidden Layers:       ", n_HL[0])
    print("Best Performing Hidden Size:         ", n_HS[0])


def mutate(ranks, n_HL, n_HS, scale, n_in):
    # Save the states of the top two networks
    hl1 = n_HL[ranks[0][1]]
    hl2 = n_HL[ranks[1][1]]
    hs1 = n_HS[ranks[0][1]]
    hs2 = n_HS[ranks[1][1]]

    # Best 2 neural networks move on
    n_HL[0], n_HS[0] = hl1, hs1
    n_HL[1], n_HS[1] = hl2, hs2

    # Mutate the best two networks
    # Scale is the iteration that it is currently in. The further along the generation is, the more refined the
    # evolutions should be
    n_HL[2] = hl1 + random.randrange(-1, 1, 1)
    n_HS[2] = hs1 + int(random.randrange(-50, 50, 1)/scale)
    n_HL[3] = hl1 + random.randrange(-1, 1, 1)
    n_HS[3] = hs1 + int(random.randrange(-50, 50, 1)/scale)
    n_HL[4] = hl1 + random.randrange(-1, 1, 1)
    n_HS[4] = hs1 + int(random.randrange(-50, 50, 1)/scale)
    n_HL[5] = hl1 + random.randrange(-1, 1, 1)
    n_HS[5] = hs1 + int(random.randrange(-50, 50, 1)/scale)
    n_HL[6] = hl2 + random.randrange(-1, 1, 1)
    n_HS[6] = hs2 + int(random.randrange(-50, 50, 1)/scale)
    n_HL[7] = hl2 + random.randrange(-1, 1, 1)
    n_HS[7] = hs2 + int(random.randrange(-50, 50, 1)/scale)

    # Crossover
    n_HL[8], n_HS[8] = hl1, hs2
    n_HL[9], n_HS[9] = hl2, hs1

    for i in range(10):
        if n_HL[i] < 0:
            n_HL[i] = 0
        if n_HS[i] < 0:
            n_HS[i] = 0

    # Create the next evolution of networks
    # CPU is used because GPU does not have enough memory to store all networks
    nn1 = NeuralNetwork(n_inputs=n_in, hidden_size=int(n_HS[0]), num_layers=int(n_HL[0]))
    GA1 = Localization(nn1, optim.Adam(nn1.parameters(), lr=learning_rate), nn.MSELoss(), nn.MSELoss())

    nn2 = NeuralNetwork(n_inputs=n_in, hidden_size=int(n_HS[1]), num_layers=int(n_HL[1]))
    GA2 = Localization(nn2, optim.Adam(nn2.parameters(), lr=learning_rate), nn.MSELoss(), nn.MSELoss())

    nn3 = NeuralNetwork(n_inputs=n_in, hidden_size=int(n_HS[2]), num_layers=int(n_HL[2]))
    GA3 = Localization(nn3, optim.Adam(nn3.parameters(), lr=learning_rate), nn.MSELoss(), nn.MSELoss())

    nn4 = NeuralNetwork(n_inputs=n_in, hidden_size=int(n_HS[3]), num_layers=int(n_HL[3]))
    GA4 = Localization(nn4, optim.Adam(nn4.parameters(), lr=learning_rate), nn.MSELoss(), nn.MSELoss())

    nn5 = NeuralNetwork(n_inputs=n_in, hidden_size=int(n_HS[4]), num_layers=int(n_HL[4]))
    GA5 = Localization(nn5, optim.Adam(nn5.parameters(), lr=learning_rate), nn.MSELoss(), nn.MSELoss())

    nn6 = NeuralNetwork(n_inputs=n_in, hidden_size=int(n_HS[5]), num_layers=int(n_HL[5]))
    GA6 = Localization(nn6, optim.Adam(nn6.parameters(), lr=learning_rate), nn.MSELoss(), nn.MSELoss())

    nn7 = NeuralNetwork(n_inputs=n_in, hidden_size=int(n_HS[6]), num_layers=int(n_HL[6]))
    GA7 = Localization(nn7, optim.Adam(nn7.parameters(), lr=learning_rate), nn.MSELoss(), nn.MSELoss())

    nn8 = NeuralNetwork(n_inputs=n_in, hidden_size=int(n_HS[7]), num_layers=int(n_HL[7]))
    GA8 = Localization(nn8, optim.Adam(nn8.parameters(), lr=learning_rate), nn.MSELoss(), nn.MSELoss())

    nn9 = NeuralNetwork(n_inputs=n_in, hidden_size=int(n_HS[8]), num_layers=int(n_HL[8]))
    GA9 = Localization(nn9, optim.Adam(nn9.parameters(), lr=learning_rate), nn.MSELoss(), nn.MSELoss())

    nn10 = NeuralNetwork(n_inputs=n_in, hidden_size=int(n_HS[9]), num_layers=int(n_HL[9]))
    GA10 = Localization(nn10, optim.Adam(nn10.parameters(), lr=learning_rate), nn.MSELoss(), nn.MSELoss())

    return GA1, GA2, GA3, GA4, GA5, GA6, GA7, GA8, GA9, GA10, n_HL, n_HS


if __name__ == "__main__":
    main()
