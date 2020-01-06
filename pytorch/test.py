import torch
from torch import nn
from collections import OrderedDict
import torch.nn.functional as F 
import torchvision

model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 1))
# print("model: ",model,type(model)) 

# print(type(model.parameters()),type(model.named_parameters()))
pa = [i for i in model.parameters()]
# print(pa)
# print(pa[0],type(pa[0]),pa[0].size())

# for named,param in model.named_parameters():
#   print(named,param.size())

class MyModel(nn.Module):
    def __init__(self, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        self.weight1 = nn.Parameter(torch.rand(20, 20))
        self.weight2 = torch.rand(20, 20)
    def forward(self, x):
        pass

n = MyModel()
for name, param in n.named_parameters():
    print(name,param.size())

