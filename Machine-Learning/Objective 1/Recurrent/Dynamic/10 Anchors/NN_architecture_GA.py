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
learning_rate = 0.00001


def main():
    start_time = time.ctime()
    os.system('clear')
    n_generations = 5
    epochs = 3

    train_no_noise = TrainLoader('training_data_NoNoise.csv')
    train_loader_no_noise = DataLoader(train_no_noise, batch_size=1, shuffle=True)

    # Create 10 nn architectures

    # Number of hidden Layers
    n_HL = torch.randint(low=2, high=5, size=(1, 10))

    # Adjust the tensor containing the number of hidden layers to a 1 dimensional tensor
    n_HL = torch.tensor([n_HL[0][0], n_HL[0][1], n_HL[0][2], n_HL[0][3], n_HL[0][4], n_HL[0][5], n_HL[0][6], n_HL[0][7],
                         n_HL[0][8], n_HL[0][9]])

    # Hidden size
    n_HS = torch.randint(low=50, high=600, size=(1, 10))

    # Adjust the tensor containing the number of neurons per layer to a 1 dimensional tensor
    n_HS = torch.tensor([n_HS[0][0], n_HS[0][1], n_HS[0][2], n_HS[0][3], n_HS[0][4], n_HS[0][5], n_HS[0][6], n_HS[0][7],
                         n_HS[0][8], n_HS[0][9]])

    # Define neural networks
    # CPU is used as the device for these networks because GPU on this computer does not have enough memory to
    # handle all of the networks.
    nn1 = NeuralNetwork(n_inputs=37, hidden_size=int(n_HS[0]), num_layers=int(n_HL[0]))
    GA1 = Localization(nn1, optim.Adam(nn1.parameters(), lr=learning_rate), nn.MSELoss(), nn.MSELoss(),
                       device=torch.device("cpu"))

    nn2 = NeuralNetwork(n_inputs=37, hidden_size=int(n_HS[1]), num_layers=int(n_HL[1]))
    GA2 = Localization(nn2, optim.Adam(nn2.parameters(), lr=learning_rate), nn.MSELoss(), nn.MSELoss(),
                       device=torch.device("cpu"))

    nn3 = NeuralNetwork(n_inputs=37, hidden_size=int(n_HS[2]), num_layers=int(n_HL[2]))
    GA3 = Localization(nn3, optim.Adam(nn3.parameters(), lr=learning_rate), nn.MSELoss(), nn.MSELoss(),
                       device=torch.device("cpu"))

    nn4 = NeuralNetwork(n_inputs=37, hidden_size=int(n_HS[3]), num_layers=int(n_HL[3]))
    GA4 = Localization(nn4, optim.Adam(nn4.parameters(), lr=learning_rate), nn.MSELoss(), nn.MSELoss(),
                       device=torch.device("cpu"))

    nn5 = NeuralNetwork(n_inputs=37, hidden_size=int(n_HS[4]), num_layers=int(n_HL[4]))
    GA5 = Localization(nn5, optim.Adam(nn5.parameters(), lr=learning_rate), nn.MSELoss(), nn.MSELoss(),
                       device=torch.device("cpu"))

    nn6 = NeuralNetwork(n_inputs=37, hidden_size=int(n_HS[5]), num_layers=int(n_HL[5]))
    GA6 = Localization(nn6, optim.Adam(nn6.parameters(), lr=learning_rate), nn.MSELoss(), nn.MSELoss(),
                       device=torch.device("cpu"))

    nn7 = NeuralNetwork(n_inputs=37, hidden_size=int(n_HS[6]), num_layers=int(n_HL[6]))
    GA7 = Localization(nn7, optim.Adam(nn7.parameters(), lr=learning_rate), nn.MSELoss(), nn.MSELoss(),
                       device=torch.device("cpu"))

    nn8 = NeuralNetwork(n_inputs=37, hidden_size=int(n_HS[7]), num_layers=int(n_HL[7]))
    GA8 = Localization(nn8, optim.Adam(nn8.parameters(), lr=learning_rate), nn.MSELoss(), nn.MSELoss(),
                       device=torch.device("cpu"))

    nn9 = NeuralNetwork(n_inputs=37, hidden_size=int(n_HS[8]), num_layers=int(n_HL[8]))
    GA9 = Localization(nn9, optim.Adam(nn9.parameters(), lr=learning_rate), nn.MSELoss(), nn.MSELoss(),
                       device=torch.device("cpu"))

    nn10 = NeuralNetwork(n_inputs=37, hidden_size=int(n_HS[9]), num_layers=int(n_HL[9]))
    GA10 = Localization(nn10, optim.Adam(nn10.parameters(), lr=learning_rate), nn.MSELoss(), nn.MSELoss(),
                        device=torch.device("cpu"))

    for i in range(n_generations):
        print("Start time:  ", start_time)
        print("Generation:  ", i+1)
        e1, e2, e3, e4, e5, e6, e7, e8, e9, e0 = [], [], [], [], [], [], [], [], [], []
        for e in range(epochs):
            a = 0
            for data, target, a1 in train_loader_no_noise:  # _no_noise:
                # Split the data between two sets of networks to speed up the training process
                pos1, loss1 = GA1.train(data, target)
                pos2, loss2 = GA2.train(data, target)
                pos3, loss3 = GA3.train(data, target)
                pos4, loss4 = GA4.train(data, target)
                pos5, loss5 = GA5.train(data, target)
                pos6, loss6 = GA6.train(data, target)
                pos7, loss7 = GA7.train(data, target)
                pos8, loss8 = GA8.train(data, target)
                pos9, loss9 = GA9.train(data, target)
                pos10, loss10 = GA10.train(data, target)
                if a % 100 == 0:
                    e1.append(loss1)
                    e2.append(loss2)
                    e3.append(loss3)
                    e4.append(loss4)
                    e5.append(loss5)
                    e6.append(loss6)
                    e7.append(loss7)
                    e8.append(loss8)
                    e9.append(loss9)
                    e0.append(loss10)

                a += 1

        # Average Error
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

        # Put all the final losses along with a corresponding number into a list of tuples
        rank = [(ae1, 0),
                (ae2, 1),
                (ae3, 2),
                (ae4, 3),
                (ae5, 4),
                (ae6, 5),
                (ae7, 6),
                (ae8, 7),
                (ae9, 8),
                (ae0, 9)]

        # Sort the list with the lowest errors appearing earlier in the array
        rank.sort()
        GA1, GA2, GA3, GA4, GA5, GA6, GA7, GA8, GA9, GA10, n_HL, n_HS = mutate(rank, n_HL, n_HS, i+1)

        os.system('clear')
        print("Best Architecture:")
        print("Hidden Layers:                  ", int(n_HL[0]))
        print("Hidden Size:                    ", int(n_HS[0]))
        print("Accuracy:                       ", float(rank[0][0]), "\n")

    os.system('clear')
    end_time = time.ctime()
    print("Start:   ", start_time)
    print("End:     ", end_time)
    print("Best Performing Hidden Layers:       ", n_HL[0])
    print("Best Performing Hidden Size:         ", n_HS[0])


def mutate(ranks, n_HL, n_HS, scale):
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
    nn1 = NeuralNetwork(n_inputs=37, hidden_size=int(n_HS[0]), num_layers=int(n_HL[0]))
    GA1 = Localization(nn1, optim.Adam(nn1.parameters(), lr=learning_rate), nn.MSELoss(), nn.MSELoss(),
                       device=torch.device("cpu"))

    nn2 = NeuralNetwork(n_inputs=37, hidden_size=int(n_HS[1]), num_layers=int(n_HL[1]))
    GA2 = Localization(nn2, optim.Adam(nn2.parameters(), lr=learning_rate), nn.MSELoss(), nn.MSELoss(),
                       device=torch.device("cpu"))

    nn3 = NeuralNetwork(n_inputs=37, hidden_size=int(n_HS[2]), num_layers=int(n_HL[2]))
    GA3 = Localization(nn3, optim.Adam(nn3.parameters(), lr=learning_rate), nn.MSELoss(), nn.MSELoss(),
                       device=torch.device("cpu"))

    nn4 = NeuralNetwork(n_inputs=37, hidden_size=int(n_HS[3]), num_layers=int(n_HL[3]))
    GA4 = Localization(nn4, optim.Adam(nn4.parameters(), lr=learning_rate), nn.MSELoss(), nn.MSELoss(),
                       device=torch.device("cpu"))

    nn5 = NeuralNetwork(n_inputs=37, hidden_size=int(n_HS[4]), num_layers=int(n_HL[4]))
    GA5 = Localization(nn5, optim.Adam(nn5.parameters(), lr=learning_rate), nn.MSELoss(), nn.MSELoss(),
                       device=torch.device("cpu"))

    nn6 = NeuralNetwork(n_inputs=37, hidden_size=int(n_HS[5]), num_layers=int(n_HL[5]))
    GA6 = Localization(nn6, optim.Adam(nn6.parameters(), lr=learning_rate), nn.MSELoss(), nn.MSELoss(),
                       device=torch.device("cpu"))

    nn7 = NeuralNetwork(n_inputs=37, hidden_size=int(n_HS[6]), num_layers=int(n_HL[6]))
    GA7 = Localization(nn7, optim.Adam(nn7.parameters(), lr=learning_rate), nn.MSELoss(), nn.MSELoss(),
                       device=torch.device("cpu"))

    nn8 = NeuralNetwork(n_inputs=37, hidden_size=int(n_HS[7]), num_layers=int(n_HL[7]))
    GA8 = Localization(nn8, optim.Adam(nn8.parameters(), lr=learning_rate), nn.MSELoss(), nn.MSELoss(),
                       device=torch.device("cpu"))

    nn9 = NeuralNetwork(n_inputs=37, hidden_size=int(n_HS[8]), num_layers=int(n_HL[8]))
    GA9 = Localization(nn9, optim.Adam(nn9.parameters(), lr=learning_rate), nn.MSELoss(), nn.MSELoss(),
                       device=torch.device("cpu"))

    nn10 = NeuralNetwork(n_inputs=37, hidden_size=int(n_HS[9]), num_layers=int(n_HL[9]))
    GA10 = Localization(nn10, optim.Adam(nn10.parameters(), lr=learning_rate), nn.MSELoss(), nn.MSELoss(),
                        device=torch.device("cpu"))

    return GA1, GA2, GA3, GA4, GA5, GA6, GA7, GA8, GA9, GA10, n_HL, n_HS


if __name__ == "__main__":
    main()
