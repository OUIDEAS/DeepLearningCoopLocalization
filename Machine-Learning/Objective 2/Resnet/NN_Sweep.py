import wandb
import pprint
import torch
import torch.nn as nn
import torch.optim as optim
from NNLib import *
import matplotlib.pyplot as plt
import os
import time
from DLoaderRI import *
from torch.utils.data import DataLoader
import sys
import json
import pickle

def train_epoch(system, optimizer, loader1, loader2):
    accv = []
    acct = []
    for data, target, a1 in loader1:
        data, target, a1 = data.to(torch.device('cuda')), target.to(torch.device('cuda')), a1.to(torch.device('cuda'))
        optimizer.zero_grad()
        output = system.Network(data)
        loss = system.Loss(output, target)
        acct.append(loss.item())
        loss.backward()
        optimizer.step()
    wandb.log({"training loss": loss.item()})

    with torch.no_grad():
        for data, target, a1 in loader2:
            data, target, a1 = data.to(torch.device('cuda')), target.to(torch.device('cuda')), a1.to(torch.device('cuda'))
            output = system.Network(data)
            loss = system.Loss(output, target)
            accv.append(loss.item())
        wandb.log({"validation_loss": loss.item()})
    return sum(accv)/len(accv), sum(acct)/len(acct)

def train(config=None):
    with wandb.init(config=config):
        config = wandb.config
        # plt.rcParams.update({'font.size': 14})
        PATH = "Localization-Network-3-Out.pt"
        print(config)
        neural_net = ResNet(size=config["fc_layer_size"], res_layers=config["layers"], residuals=config["residuals"],drop=config["drop"])
        print("************************** Created Network *************************************")
        if config["optimizer"] == "adam":
            print("ADAM")
            optimizer = optim.Adam(neural_net.parameters(), lr=config["lr"])#config["learning_rate"])
        elif config["optimizer"] =="nadam":
            print("NADAM")
            optimizer = optim.NAdam(neural_net.parameters(), lr=config["lr"])#config["learning_rate"])
        elif config["optimizer"]=="radam":
            print("RADAM")
            optimizer = optim.RAdam(neural_net.parameters(), lr=config["lr"])#config["learning_rate"])
        print("************************** Configured optimizer ********************************")
        networkclass = Localization(neural_net, optimizer, nn.L1Loss())
        print("************************** Created NN fcns class *******************************")
        acc = []
        averr = []
        N = 100
        val_acc = []
        train_acc = []
        val = TrainLoader('Validation_data.csv')
        validation_loader = DataLoader(val, batch_size=1000, shuffle=True)
        print("Validation Data Loaded")
        train = TrainLoader('Training_data.csv')
        train_loader = DataLoader(train, batch_size=config["batch_size"], shuffle=True )

        print("************************** TRAINING ********************************************")
        best_val = None
        best_train = None
        train = []
        val = []
        for e in range(config["epochs"]):
            avg_loss, training_loss = train_epoch(networkclass, optimizer, train_loader, validation_loader)
            train.append(training_loss)
            val.append(avg_loss)
            wandb.log({"Average Validation Loss":avg_loss, "epoch":e})

        open_file = open("validation_results_sweep.pkl", "rb")
        validation_accuracy = pickle.load(open_file)
        open_file.close()

        if validation_accuracy[len(validation_accuracy)-1] > val[len(val)-1]:
            write = True
        else:
            write = False

        if write:
            open_file = open("training_results_sweep.pkl", "wb")
            pickle.dump(train, open_file)
            open_file.close()

            open_file = open("validation_results_sweep.pkl", "wb")
            pickle.dump(val, open_file)
            open_file.close()



        networkclass = None
        neural_net = None

if __name__ == "__main__":
    os.system('clear')
    wandb.login()
    with open('sweep_data.json') as config_file:
        config = json.load(config_file)
        id = config["ID"]
    sweep_id = "rgeng98/Cooperative Localization NN/"+id
    wandb.agent(sweep_id, function=train, count=50)
    
    open_file = open("validation_results_sweep.pkl", "rb")
    validation_accuracy = pickle.load(open_file)
    open_file.close()
    open_file = open("training_results_sweep.pkl", "rb")
    training_acc = pickle.load(open_file)
    open_file.close()

    plt.figure()
    plt.plot(validation_accuracy, label="Validation Accuracy")
    plt.plot(training_acc, label="Training Accuracy")
    plt.legend()
    plt.xlabel("Epoch [-]")
    plt.ylabel("Mean Absoluter Error [m]")
    plt.show()
