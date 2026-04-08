import hiddenlayer as hl
import torch
model = torch.load("Localization-Network-3-Out.pt", map_location=torch.device('cpu'))

hl.build_graph(model, torch.zeros((1,204)))
