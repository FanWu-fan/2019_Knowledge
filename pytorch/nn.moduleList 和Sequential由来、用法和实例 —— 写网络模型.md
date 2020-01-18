<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#nn.moduleList-和-Sequential由来和用法" data-toc-modified-id="nn.moduleList-和-Sequential由来和用法-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>nn.moduleList 和 Sequential由来和用法</a></span><ul class="toc-item"><li><span><a href="#nn.Sequential()对象" data-toc-modified-id="nn.Sequential()对象-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>nn.Sequential()对象</a></span><ul class="toc-item"><li><span><a href="#模型建立的方法" data-toc-modified-id="模型建立的方法-1.1.1"><span class="toc-item-num">1.1.1&nbsp;&nbsp;</span>模型建立的方法</a></span></li><li><span><a href="#检查以及调用模型" data-toc-modified-id="检查以及调用模型-1.1.2"><span class="toc-item-num">1.1.2&nbsp;&nbsp;</span>检查以及调用模型</a></span></li><li><span><a href="#根据名字或者序号提取Module对象" data-toc-modified-id="根据名字或者序号提取Module对象-1.1.3"><span class="toc-item-num">1.1.3&nbsp;&nbsp;</span>根据名字或者序号提取Module对象</a></span></li></ul></li><li><span><a href="#nn.ModuleList()对象" data-toc-modified-id="nn.ModuleList()对象-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>nn.ModuleList()对象</a></span><ul class="toc-item"><li><span><a href="#extend和append方法" data-toc-modified-id="extend和append方法-1.2.1"><span class="toc-item-num">1.2.1&nbsp;&nbsp;</span>extend和append方法</a></span></li><li><span><a href="#建立以及使用方法" data-toc-modified-id="建立以及使用方法-1.2.2"><span class="toc-item-num">1.2.2&nbsp;&nbsp;</span>建立以及使用方法</a></span></li></ul></li></ul></li></ul></div>

# nn.moduleList 和 Sequential由来和用法


对于cnn前馈神经网络如果前馈一次写一个forward函数会有些麻烦，在此就有两种简化方式，<span class="burk">**ModuleLis**</span>t和<span class="burk">**Sequential**</span>。其中Sequential是一个特殊的module，它包含几个子Module，<span class="girk">前向传播时会将输入一层接一层的传递下去</span>。ModuleList也是一个特殊的module，可以包含几个子module，可以像用list一样使用它，但<span class="girk">不能直接把输入传给ModuleList</span>。下面举例说明。

## nn.Sequential()对象

建立nn.Sequential()对象，必须小心确保一个块的输出大小与下一个块的输入大小匹配。基本上，它的行为就像一个nn.Module。

### 模型建立的方法

* <span class="mark">第一种写法</span>： 
nn.Sequential()对象.add_module(层名，层class的实例）


```python
import torch
from torch import nn

net1 = nn.Sequential()
net1.add_module('conv',nn.Conv2d(3,3,3))
net1.add_module('batchnorm',nn.BatchNorm2d(3))
net1.add_module('activation_layer',nn.ReLU())
```

* <span class="mark">第二种写法</span>(*多个层class的实例）


```python
net2 = nn.Sequential(
nn.Conv2d(3,3,3),
nn.BatchNorm2d(3),
nn.ReLU())
```

* <span class="mark">第三种写法</span> ： nn.Sequential(OrderDict([*多个(层名，层class的实例）]))


```python
from collections import OrderedDict
net3 = nn.Sequential(OrderedDict([
    ('conv',nn.Conv2d(3,3,3)),
    ('batchnorm',nn.BatchNorm2d(3)),
    ('activation_layer',nn.ReLU())
]))
```

### 检查以及调用模型


```python
print('net1: ',net1)
print('net2: ',net2)
print('net3: ',net3)
```

    net1:  Sequential(
      (conv): Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1))
      (batchnorm): BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (activation_layer): ReLU()
    )
    net2:  Sequential(
      (0): Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1))
      (1): BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
    )
    net3:  Sequential(
      (conv): Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1))
      (batchnorm): BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (activation_layer): ReLU()
    )
    

### 根据名字或者序号提取Module对象


```python
net1.conv,net2[0],net3.conv
```




    (Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1)),
     Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1)),
     Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1)))



## nn.ModuleList()对象

**为何有他？**
写一个module然后就写foreword函数很麻烦，所以就有了这两个。它被<span class="burk">设计用来存储任意数量</span>的nn.module

**什么时候用？**

如果在构造函数__init__中用到list、tuple、dict等对象时，一定要思考是否应该用ModuleList或ParameterList代替。

如果你想设计一个神经网络的层数作为输入传递

**和List之间区别**

ModuleList是Module的子类，当在Module中使用它的时候，就能自动识别为子module。
当添加 nn.ModuleList 作为 nn.Module 对象的一个成员时（即当我们添加模块到我们的网络时），所有 nn.ModuleList 内部的 nn.Module 的 parameter 也被添加作为 我们的网络的 parameter。


```python
class MyModule(nn.Module):
    def __init__(self):
        super(MyModule,self).__init__()
        self.linears = nn.ModuleList([nn.Linear(10,10) for i in range(10)])
    def forward(self,x):
        # ModuleList can act as an iterable, or be indexed
        for i,l in enumerate(self.linears):
            x = self.linears[i//2](x) + l(x)
            return x
```


```python
net = MyModule()
print(net)
```

    MyModule(
      (linears): ModuleList(
        (0): Linear(in_features=10, out_features=10, bias=True)
        (1): Linear(in_features=10, out_features=10, bias=True)
        (2): Linear(in_features=10, out_features=10, bias=True)
        (3): Linear(in_features=10, out_features=10, bias=True)
        (4): Linear(in_features=10, out_features=10, bias=True)
        (5): Linear(in_features=10, out_features=10, bias=True)
        (6): Linear(in_features=10, out_features=10, bias=True)
        (7): Linear(in_features=10, out_features=10, bias=True)
        (8): Linear(in_features=10, out_features=10, bias=True)
        (9): Linear(in_features=10, out_features=10, bias=True)
      )
    )
    

### extend和append方法

<span class="burk">nn.moduleList定义对象后，有extend和append方法</span>，用法和python中一样，<span class="burk">extend是添加另一个modulelist</span>  <span class="burk">append是添加另一个module</span>


```python
class LinearNet(nn.Module):
  def __init__(self, input_size, num_layers, layers_size, output_size):
     super(LinearNet, self).__init__()
 
     self.linears = nn.ModuleList([nn.Linear(input_size, layers_size)])
     self.linears.extend([nn.Linear(layers_size, layers_size) for i in range(1, self.num_layers-1)])
     self.linears.append(nn.Linear(layers_size, output_size))
```

### 建立以及使用方法


```python
modellist = nn.ModuleList
```
