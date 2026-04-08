import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import numpy as np

class ResBlock(nn.Module):
    def __init__(self, res_layers, size, drop, residuals):
        super().__init__()
        self.layers = res_layers*residuals
        rb = []
        for i in range(res_layers):
            rb.append(nn.Linear(size, size))
            rb.append(nn.Dropout(drop))
            rb.append(nn.PReLU(num_parameters=size))
        self.res = nn.Sequential(*rb)
        self.apply(self.rg_init)

    def forward(self, x):
        return x + self.res(x)

    def rg_init(self, m):
        if isinstance(m, torch.nn.Linear):
            (fan_in, fan_out) = nn.init._calculate_fan_in_and_fan_out(m.weight)
            c = 1
            var = c/(self.layers*fan_in)
            nn.init.normal_(m.weight, 0, math.sqrt(var))
            m.bias.data.zero_()

class ResNet(nn.Module):
    
    def __init__(self, n_in:int, size:int, res_layers:int, residuals:int, drop=0.1):
        super().__init__()
        self.in_layer = nn.Linear(n_in, size)
        self.activation = nn.PReLU(num_parameters=size)
        self.ResidualNetwork = nn.ModuleList([ResBlock(res_layers, size, drop, residuals) for i in range(residuals)])
        self.out = nn.Linear(size, 3)

    def forward(self, x):
        x = self.in_layer(x)
        x = self.activation(x)
        for layer in self.ResidualNetwork:
            x = layer(x)
        x = self.out(x)
        return x


class DataStandardizer():

    def __init__(self):
        pass

    def clear(self):
      self.input_batch = None
      self.devs = []
      self.means = []

    def StandardizeInputs(self, input_batch, List = False):
        if not isinstance(input_batch, list):
            self.input_batch = input_batch.cpu().numpy()
            single_state = False
        else:
            self.input_batch = input_batch
            single_state = True

        self.devs = []
        self.means = []
        new_inputs = []
        if not single_state:
            for inputs in self.input_batch:
                stdev = self.standard_dev(inputs)
                self.devs.append(stdev)
                mean = sum(inputs)/len(inputs)
                self.means.append(mean)
                new_inputs.append([(i-mean)/stdev for i in inputs])
        
        else:
            stdev = self.standard_dev(input_batch)
            self.devs = stdev
            mean = sum(input_batch)/len(input_batch)
            self.means = mean
            new_inputs = [(i-mean)/stdev for i in input_batch]

        if List:
            return list(new_inputs)
        else:
            return torch.tensor(new_inputs).to(torch.float32)

    def StandardizeTargets(self, targets, List = False):
        if not isinstance(targets, list):
            target_batch = targets.cpu().numpy()
            single_state = False
        else:
            self.input_batch = targets
            single_state = True

        new_targets = []
        batch = 0
        if not single_state:
            for t in target_batch:
                new_targets.append([(i - self.means[batch])/self.devs[batch] for i in t])
                batch += 1
        else:
            new_targets = [(i - self.means)/self.devs for i in targets]
        if List:
            return new_targets
        else:
            return torch.tensor(new_targets)

    def standard_dev(self, input):
        mean = sum(input)/len(input)
        var = sum((i - mean)**2 for i in input)/(len(input))
        return np.sqrt(var)
    
    def InverseStandardize(self, outputs):
        pos = outputs.cpu().detach().numpy()
        index = 0
        new_outputs = []
        for p in pos:
            n = []
            for i in p:
                n.append(i*self.devs[index] + self.means[index])
                i *= self.devs[index]
                i += self.means[index]
            new_outputs.append(n)
            index += 1

        return torch.tensor(new_outputs)



# Class that handles all neural network operations
class Localization(object):

    def __init__(self, network, optimizer, criterion1,
                 device=torch.device("cuda" if torch.cuda.is_available()
                                     else "cpu")):
        self.device = device
        self.Network = network.to(self.device)
        self.Optim = optimizer
        self.Loss = criterion1
        self.standardizer = DataStandardizer()
        self.x = None
        self.y = None
        self.z = None

    def train(self, inputs, targets, scaler = None):
        if scaler is not None:
            inputs = self.standardizer.StandardizeInputs(inputs)
            targets_ = self.standardizer.StandardizeTargets(targets)
        inputs = inputs.to(self.device)
        t = targets.to(self.device)
        self.Optim.zero_grad()
        pos = self.Network(inputs)
        if scaler is not None:
            error = self.Loss(pos, targets_.to(self.device))
            pos = self.standardizer.InverseStandardize(pos)
            error.backward()
            error = self.Loss(pos.to(self.device), t)

        else:
            error = self.Loss(pos, Variable(t))
            error.backward()

        self.Optim.step()

        return pos, error.item()


    def test(self, inputs, targets, scaler = None):
        if scaler is not None:
            inputs = self.standardizer.StandardizeInputs(inputs)
            
        inputs = inputs.to(self.device)
        t = targets.to(self.device)
        
        with torch.no_grad():
            pos = self.Network(inputs)
            if scaler is not None:
                pos = self.standardizer.InverseStandardize(pos).to(self.device)
                pos = pos.to(self.device)
            error = self.Loss(pos, t)
            return pos, error.item()

    def deploy(self, inputs, a1):
        # inputs = self.standardizer.StandardizeInputs(inputs)
        inputs = inputs.to(self.device)
        with torch.no_grad():
            pos = self.Network(inputs)
            # pos = self.standardizer.InverseStandardize(pos)
            self.x = float(pos[0][0]) + float(a1[0][0])
            self.y = float(pos[0][1]) + float(a1[0][1])
            self.z = float(pos[0][2]) + float(a1[0][2])
        return pos

