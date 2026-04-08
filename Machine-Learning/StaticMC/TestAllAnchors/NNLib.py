import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable
import numpy as np
import math
from torch import Tensor

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

class GainNet(nn.Module):
    def __init__(self, size:int, res_layers:int = 2, residuals:int = 2, drop=0.1):
        super().__init__()
        self.in_layer = nn.Linear(222+6, size)
        self.activation = nn.PReLU(num_parameters=size)
        self.ResidualNetwork = nn.ModuleList([ResBlock(res_layers, size, drop, residuals) for i in range(residuals)])
        self.out = nn.Linear(size, 3)
    def forward(self, data, INS_Estimate, ANN_Estimate):
        x = torch.cat((data, INS_Estimate, ANN_Estimate), 1)
        x = self.in_layer(x)
        x = self.activation(x)
        for layer in self.ResidualNetwork:
            x = layer(x)
        x = nn.Sigmoid(self.out(x))

        position = torch.tensor([[INS_Estimate[0][0]*(1-x[0][0]) + ANN_Estimate*(x[0][0]),
                                  INS_Estimate[0][1]*(1-x[0][1]) + ANN_Estimate*(x[0][1]),
                                  INS_Estimate[0][2]*(1-x[0][2]) + ANN_Estimate*(x[0][2])]])
        
        return position

# Initialize weights according to https://arxiv.org/abs/1502.01852
# Normal Distribution:
def initialize_weights_kn(m):
    if isinstance(m, torch.nn.Linear):
        nn.init.kaiming_normal_(m.weight,a = 0.2, mode='fan_in', nonlinearity='leaky_relu')#, gain=nn.init.calculate_gain('leaky_relu',0.2))

class FeedForward(nn.Module):
    def __init__(self, size: int, layers: int, drop = 0.1):
        super().__init__()
        network = [nn.Linear(222, size), nn.Dropout(drop), nn.PReLU(num_parameters=size)]
        for i in range(layers-1):
            network.append(nn.Linear(size, size))
            network.append(nn.Dropout(drop))
            network.append(nn.PReLU(num_parameters=size))
        network.append(nn.Linear(size, 3))
        self.Network = nn.Sequential(*network)

    def forward(self, x):
        return self.network(x)

# Uniform distribution:
def initialize_weights_ku(m):
    if isinstance(m, torch.nn.Linear):
        nn.init.kaiming_uniform_(m.weight,a = 0.2, mode='fan_in', nonlinearity='leaky_relu')#, gain=nn.init.calculate_gain('leaky_relu',0.2))

# Initialize weights according to https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
# Normal Distribution:
def initialize_weights_xn(m):
    if isinstance(m, torch.nn.Linear):
        nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('leaky_relu',0.2))

# Uniform distribution
def initialize_weights_xu(m):
    if isinstance(m, torch.nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('leaky_relu',0.2))

# Initialize weights according to https://openreview.net/forum?id=_wzZwKpTDF_9C
def initialize_weights_orth_(m):
    if isinstance(m, torch.nn.Linear):
        nn.init.orthogonal(m.weight)


class Net3(torch.nn.Module):
    def __init__(self, n_inputs, shared_size, shared_layers, hidden_size, num_layers):
        super(Net3, self).__init__()
        self.cer = True
        self.n_inputs = n_inputs
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.shared_size = shared_size

        sl = [torch.nn.Linear(self.n_inputs, self.shared_size)]

        for i in range(shared_layers):
            sl.append(torch.nn.LeakyReLU(negative_slope=0.2))
            sl.append(torch.nn.Linear(self.shared_size, self.shared_size))

        sl.append(torch.nn.LeakyReLU(negative_slope=0.2))

        self.shared = torch.nn.Sequential(*sl)

        hidden_layers_e = [torch.nn.Linear(self.shared_size, self.hidden_size)]
        hidden_layers_n = [torch.nn.Linear(self.shared_size, self.hidden_size)]
        hidden_layers_u = [torch.nn.Linear(self.shared_size, self.hidden_size)]

        # First hidden layer is defined separately because number of inputs to that layer is different than the rest
        for i in range(self.num_layers):
            # Add a fully connected layer and then a Hyperbolic
            # tangent activation function for each layer that is added
            hidden_layers_e.append(torch.nn.LeakyReLU(negative_slope=0.2))
            hidden_layers_e.append(torch.nn.Linear(self.hidden_size, self.hidden_size))

            hidden_layers_n.append(torch.nn.LeakyReLU(negative_slope=0.2))
            hidden_layers_n.append(torch.nn.Linear(self.hidden_size, self.hidden_size))

            hidden_layers_u.append(torch.nn.LeakyReLU(negative_slope=0.2))
            hidden_layers_u.append(torch.nn.Linear(self.hidden_size, self.hidden_size))


        hidden_layers_e.append(torch.nn.LeakyReLU(negative_slope=0.2))
        hidden_layers_u.append(torch.nn.LeakyReLU(negative_slope=0.2))
        hidden_layers_n.append(torch.nn.LeakyReLU(negative_slope=0.2))

        hidden_layers_e.append(torch.nn.Linear(self.hidden_size, 1))
        hidden_layers_n.append(torch.nn.Linear(self.hidden_size, 1))
        hidden_layers_u.append(torch.nn.Linear(self.hidden_size, 1))

        self.NetEast = torch.nn.Sequential(*hidden_layers_e)
        self.NetNorth = torch.nn.Sequential(*hidden_layers_n)
        self.NetUp = torch.nn.Sequential(*hidden_layers_u)



    def forward(self, x):
        x = self.shared(x)
        e = self.NetEast(x)
        n = self.NetNorth(x)
        u = self.NetUp(x)

        return e, n, u

# Class that handles all neural network operations
class Localization(object):

    def __init__(self, network, optimizer, criterion1,
                 device=torch.device("cuda" if torch.cuda.is_available()
                                     else "cpu")):
        self.device = device
        self.Network = network.to(self.device)
        self.Optim = optimizer
        self.Loss = criterion1
        self.x = None
        self.y = None
        self.z = None

    def train(self, inputs, targets):
        # if self.Network.cer:
        #     indicese = torch.zeros([targets.size(dim=-2), 1], dtype=torch.int64)
        #     indicesn = torch.ones([targets.size(dim=-2), 1], dtype=torch.int64)
        #     indicesu = 2*torch.ones([targets.size(dim=-2), 1], dtype=torch.int64)
        #     te = targets.gather(1,indicese)
        #     tn = targets.gather(1,indicesn)
        #     tu = targets.gather(1,indicesu)
        #     inputs, te, tn, tu = inputs.to(self.device), te.to(self.device), tn.to(self.device), tu.to(self.device)
        #     self.Optim.zero_grad()
        #     east, north, up = self.Network(Variable(inputs))
        #     errore = self.Loss(east, Variable(te))
        #     errorn = self.Loss(north, Variable(tn))
        #     erroru = self.Loss(up, Variable(tu))
        #     errore.backward(retain_graph = True)
        #     errorn.backward(retain_graph = True)
        #     erroru.backward(retain_graph = True)
        #     self.Optim.step()
        #     pos = torch.tensor([[east[0][0], north[0][0], up[0][0]]])
        #     mae = (errore.item() + errorn.item() + erroru.item())/3
        #     return pos, mae
        # else:
        inputs, t = inputs.to(self.device), targets.to(self.device)
        self.Optim.zero_grad()
        pos = self.Network(Variable(inputs))
        error = self.Loss(pos, Variable(t))
        error.backward()
        self.Optim.step()
        return pos, error.item()


    def test(self, inputs, targets):
        if self.Network.cer:
            indicese = torch.zeros([targets.size(dim=-2), 1], dtype=torch.int64)
            indicesn = torch.ones([targets.size(dim=-2), 1], dtype=torch.int64)
            indicesu = 2*torch.ones([targets.size(dim=-2), 1], dtype=torch.int64)
            te = targets.gather(1,indicese)
            tn = targets.gather(1,indicesn)
            tu = targets.gather(1,indicesu)
            inputs, te, tn, tu = inputs.to(self.device), te.to(self.device), tn.to(self.device), tu.to(self.device)
            east, north, up = self.Network(inputs)
            errore = self.Loss(east, Variable(te))
            errorn = self.Loss(north, Variable(tn))
            erroru = self.Loss(up, Variable(tu))
            error = errore + errorn + erroru
            pos = torch.tensor([[east[0][0], north[0][0], up[0][0]]])
            mae = (errore.item() + errorn.item() + erroru.item())/3
            return pos, mae
        else:
            inputs, t = inputs.to(self.device), targets.to(self.device)
            with torch.no_grad():
                pos = self.Network(inputs)
                error1 = self.Loss(pos, t)
                error = error1
                return pos, error.item()

    def deploy(self, inputs, a1):
        if self.Network.cer:
            inputs = inputs.to(self.device)
            east, north, up = self.Network(Variable(inputs))
            self.x = float(east[0][0]) + float(a1[0][0])
            self.y = float(north[0][0]) + float(a1[0][1])
            self.z = float(up[0][0]) + float(a1[0][2])
            pos = torch.tensor([[east[0][0], north[0][0], up[0][0]]])
            return pos
        else:
            inputs = inputs.to(self.device)
            with torch.no_grad():
                pos = self.Network(inputs)
                self.x = float(pos[0][0]) + float(a1[0][0])
                self.y = float(pos[0][1]) + float(a1[0][1])
                self.z = float(pos[0][2]) + float(a1[0][2])
            return pos


    @staticmethod
    def plot3ax(x, y, z, xnn, ynn, znn):
        fig = plt.figure(1)
        plt.clf()
        ax = plt.axes(projection='3d')
        ax.set_xlim3d(-10, 285)
        ax.set_ylim3d(-10, 285)
        ax.set_zlim3d(-10, 285)
        plt.title('Feed Forward Dynamic Anchor Test - Sliding Window with Noise')
        ax.plot3D(x, y, z, color='gray')
        ax.scatter(xnn, ynn, znn, color='green')
        plt.pause(0.001)

class Node(nn.Module):
    def __init__(self, res_layers, size, drop, residuals):
        super().__init__()
        self.layers = res_layers*residuals
        rb = []
        for i in range(res_layers):
            rb.append(nn.Linear(size, size))
            rb.append(nn.Dropout(0.1))
            rb.append(nn.PReLU(num_parameters=size))
        self.res = nn.Sequential(*rb)
        self.apply(self.rg_init)

    def forward(self, x):
        return self.res(x)

    def rg_init(self, m):
        if isinstance(m, torch.nn.Linear):
            (fan_in, fan_out) = nn.init._calculate_fan_in_and_fan_out(m.weight)
            c = 1
            var = c/(self.layers*fan_in)
            nn.init.normal_(m.weight, 0, math.sqrt(var))
            m.bias.data.zero_()

class ResNeXt(nn.Module):
    def __init__(self, size:int, res_layers:int, residuals:int, drop=0.1):
        super().__init__()
        self.in_layer = nn.Linear(222, size)
        self.activation = nn.PReLU(num_parameters=size)
        self.ResidualNetwork = nn.ModuleList([ResBlock(res_layers, size, drop, residuals) for i in range(residuals)])
        self.out = nn.Linear(size, 3)
    def forward(self, x):
        x = self.in_layer(x)
        x = self.activation(x)
        resids = []
        for layer in self.ResidualNetwork:
            resids.append(layer(x))

        out=torch.zeros(x.size()).to(torch.device('cuda'))
        for r in resids:
            out = out + r
        x = self.out(x)
        return x

class NodalLayer(nn.Module):
    def __init__(self, nodes, size, layers, drop):
        super().__init__()
        self.Nodes = nn.ModuleList([Node(layers, size, drop, nodes) for i in range(nodes)])
    def forward(self, x):
        resids=[]
        for layer in self.Nodes:
            resids.append(layer(x))

        out=torch.zeros(x.size()).to(torch.device('cuda'))
        for r in resids:
            out = out + r
        return out+x

class RGNet(nn.Module):
    def __init__(self, size:int, res_layers:int, nodes:int, nodal_layers:int, drop=0.1):
        super().__init__()
        self.in_layer = nn.Linear(222, size)
        self.activation = nn.PReLU(num_parameters=size)
        self.Network = nn.ModuleList([NodalLayer(nodes, size, nodal_layers, drop) for i in range(nodal_layers)])
        self.out = nn.Linear(size, 3)

    def forward(self, x):
        x = self.in_layer(x)
        x = self.activation(x)
        resids = []
        # First Cluster of SubNetworks
        for layer in self.Network:
            x = layer(x)

        x = self.out(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, res_layers, size, drop, residuals):
        super().__init__()
        self.layers = res_layers*residuals
        rb = []
        for i in range(res_layers):
            rb.append(nn.Linear(size, size))
            rb.append(nn.Dropout(0.1))
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
    def __init__(self, size:int, res_layers:int, residuals:int, drop=0.1):
        super().__init__()
        self.in_layer = nn.Linear(222, size)
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

class EKF():
    def __init__(self, state, dt):
        # State = [x, y, z, vx, vy, vz]
        # Inputs = [ax, ay, az]
        self.x_k = state
        self.x = self.x_k
        self.dt = dt
        self.A = np.matrix([[1, 0, 0, dt, 0, 0, 1/2*dt**2, 0, 0],
                            [0, 1, 0, 0, dt, 0, 0, 1/2*dt**2, 0],
                            [0, 0, 1, 0, 0, dt, 0, 0, 1/2*dt**2],
                            [0, 0, 0, 1, 0, 0, dt, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, dt, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0, dt],
                            [0, 0, 0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 1]])
        
        self.Q = np.eye(9)*0.1
        self.P = np.eye(9)

    def predict(self, acc):
        self.x_prev = np.copy(self.x_k)
        self.P_prev = np.copy(self.P)
        x = self.x_k
        x[6] = acc[0]
        x[7] = acc[1]
        x[8] = acc[2]
        self.x = self.A*x
        self.P = self.A*self.P*self.A.T + self.Q

    def format_data(self, ancs):
        rho = []
        points = [[0, 0, 0]]
        for p in ancs:
            rho.append(p[3])
        
        for i in range(len(ancs)-1):
            points.append([ancs[i+1][0] - ancs[0][0],
                           ancs[i+1][1] - ancs[0][1],
                           ancs[i+1][2] - ancs[0][2]])

        return np.array(points), np.array(rho)

    def observe(self, Ancs):
        self.SensorError(Ancs)
        Skip = True
        
        # Create a Column Matrix of Ranges
        for p in Ancs:
            if Skip:
                self.D = np.matrix([[float(Ancs[0][3])]], dtype = np.double)
                Skip = False
            else:
                self.D = np.append(self.D,[[float(p[3])]], axis = 0)

        Skip = True


        # Create the Jacobian Matrix for the Observation Step
        points, rho = self.format_data(Ancs)
        # Initial guess
        u = np.array([[0],[0],[0]])
        # Ordinary Least Squares Solution
        for i in range(100):
            # Create G matrix
            G = np.array([[float((points[0][0]-u[0])/rho[0]),  float((points[0][1]-u[1])/rho[0]),  float((points[0][2]-u[2])/rho[0]) ]])
            for i in range(len(points)-2):
                row = np.array([float((points[i+1][0]-u[0])/rho[i+1]),  float((points[i+1][1]-u[1])/rho[i+1]),  float((points[i+1][2]-u[2])/rho[i+1]) ])
                G = np.append(G, [row], axis=0)

            # Create dRho matrix
            rho_hat = []
            for p in range(len(points)-1):
                r = float(np.sqrt((points[p][0]-u[0])**2 + (points[p][1]-u[1])**2 + (points[p][2]-u[2])**2))
                rho_hat.append(r)
                if p == 0:
                    drho = np.array([[float(rho_hat[0]-rho[0])]])
                else:
                    dr = np.array([float(rho_hat[p]-rho[p])])
                    drho = np.append(drho, [dr], axis=0)
            # Linear Algebra to adjust the position estimate
            a = np.matmul(np.transpose(G),G)
            du = np.matmul(np.matmul(np.linalg.inv(a), np.transpose(G)), drho)
            # Update estimate
            u = u + du
        u[0] = u[0] + Ancs[0][0]
        u[1] = u[1] + Ancs[0][1]
        u[2] = u[2] + Ancs[0][2]

        for p in Ancs:
            if Skip:
                r = float(np.sqrt((u[0] - p[0])**2 + (u[1] - p[1])**2 + (u[2] - p[2])**2))
                pdx = float((p[0] - u[0])/p[3])
                pdy = float((p[1] - u[1])/p[3])
                pdz = float((p[2] - u[2])/p[3])
                self.H = np.matrix([[float(pdx), float(pdy), float(pdz), 0, 0, 0, 0, 0, 0]], dtype = np.double)
                G = np.array([[float(pdx), float(pdy), float(pdz)]], dtype = np.double)
                self.rho = np.array([[r]], dtype = np.double)
                self.z = np.array([p[3]], dtype = np.double)
                Skip = False
            else:
                r = float(np.sqrt((u[0] - p[0])**2 + (u[1] - p[1])**2 + (u[2] - p[2])**2))
                pdx = (p[0] - u[0])/p[3]
                pdy = (p[1] - u[1])/p[3]
                pdz = (p[2] - u[2])/p[3]
                self.H = np.append(self.H, [[float(pdx), float(pdy), float(pdz), 0, 0, 0, 0, 0, 0]], axis = 0)
                row = np.array([float(pdx), float(pdy), float(pdz)])
                G = np.append(G, [row], axis = 0)
                self.rho = np.append(self.rho, np.array([[r]]), axis = 0)
                self.z = np.append(self.z, np.array([p[3]]), axis = 0)
        self.z = np.transpose(np.array([self.z]))
        
        self.x_update = self.x
        self.x_update[0] = u[0]
        self.x_update[1] = u[1]
        self.x_update[2] = u[2]
                

    def update(self):
        self.Kalman = self.P*self.H.T*np.linalg.inv(self.H*self.P*self.H.T+self.R)
        residual = np.subtract
        y = residual(self.z, self.rho)
        self.x_k = self.x + self.Kalman * y
        self.P = (np.eye((self.Kalman*self.H).shape[1]) - self.Kalman*self.H)*self.P

    def SensorError(self, Ancs):
        b = 3.1*10**9
        c = 3e8
        self.R = np.eye(len(Ancs))
        # for i in range(len(Ancs)):
        #     try:
        #         r = Ancs[i][2]
        #         snr = 10 - r*(10-0.1)/600
        #         sigsq = 1/(8*(math.pi**2)*(b**2)*snr)
        #         sig = 40*c*math.sqrt(sigsq)
        #         r_err = float(sig*sig)
        #         self.R[i][i] = sig
        #     except:
        #         snr = 0.00000000000000000000000001
        #         sigsq = 1/(8*(math.pi**2)*(b**2)*snr)
        #         sig = 40*c*math.sqrt(sigsq)
        #         r_err = float(sig*sig)
        #         self.R[i][i] = sig
