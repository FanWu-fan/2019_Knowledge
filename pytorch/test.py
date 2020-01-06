import torch
from torch import nn
from collections import OrderedDict
import torch.nn.functional as F 
import torchvision
from  torchsummary  import summary



# model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 1))
# print("model: ",list(model.modules())) 
# print(model)

# print(type(model.parameters()),type(model.named_parameters()))
# pa = [i for i in model.parameters()]
# print(pa)
# print(pa[0],type(pa[0]),pa[0].size())

# for named,param in model.named_parameters():
#   print(named,param.size())

# print(model.state_dict(),type(model.state_dict()))
# print(model.state_dict()['0.weight'].numel(),model.state_dict()['0.weight'])

# for name,param in model.state_dict().items():
#     print(name,param) #,param.requires_grad=True)



# model = torchvision.models.vgg16(pretrained=False)
# # summary(model,(3,224,224))
# print(list(model.children()))

# class Net(nn.Module):
#     def __init__(self,model):
#         super(Net,self).__init__()
#         # -2 表示去掉model的最后两层
#         self.vgg_layer = nn.Sequential(*list(model.children())[:-2])
#         self.transion_layer = nn.ConvTranspose2d(2048,2048,kernel_size=12,stride=3)
#         self.pool_layer = nn.MaxPool2d(32)
#         self.Linear_layer = nn.Linear(2048,8)
    
#     def forward(self, x):
#         x = self.vgg_layer(x)
#         x = self.transion_layer(X)
#         x = self.pool_layer(x)
#         # 将一个多行的Tensor,拼接成为一行，-1指自适应
#         x = x.view(x.size(0),-1)
#         x = self.Linear_layer(x)
#         return x 

# mymodel = Net(model)
# print('-'*30)
# print(list(mymodel.children()))


model = torchvision.models.densenet121(pretrained=False)
# summary(model,(3,224,224))
model.features.conv0.in_channels = 6 
# print(model.features.conv0.in_channels)
# print(model)
model.features.conv0 = nn.Conv2d(6, 64, kernel_size=7, stride=2,
                                padding=3, bias=False)

# summary(model,(6,299,299))



# class Net(nn.Module):
#     def __init__(self,model):
#         super(Net,self).__init__()
#         # -2 表示去掉model的最后两层
#         self.vgg_layer = nn.Sequential(*list(model.children())[:-2])
#         self.transion_layer = nn.ConvTranspose2d(2048,2048,kernel_size=12,stride=3)
#         self.pool_layer = nn.MaxPool2d(32)
#         self.Linear_layer = nn.Linear(2048,8)
    
#     def forward(self, x):
#         x = self.vgg_layer(x)
#         x = self.transion_layer(X)
#         x = self.pool_layer(x)
#         # 将一个多行的Tensor,拼接成为一行，-1指自适应
#         x = x.view(x.size(0),-1)
#         x = self.Linear_layer(x)
#         return x 

# mymodel = Net(model)
# print('-'*50)
# print(list(mymodel.children()))