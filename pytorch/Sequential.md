# nn.Sequential

## __call__方法

关于 *\_\_call__* 方法，不得不先提到一个概念，就是可调用对象（callable），我们平时**自定义的函数**、**内置函数**和**类**都属于可调用对象，但凡是可以**把一对括号()应用到某个对象身上**都可称之为可调用对象，判断对象是否为可调用对象可以用函数 **callable**

如果在类中实现了 *\_\_call__* 方法，那么实例对象也将成为一个可调用对象，

在Python中，方法也是一种高等的对象。这意味着他们也可以被传递到方法中就像其他对象一样。这是一个非常惊人的特性。 在Python中，**一个特殊的魔术方法可以让类的实例的行为表现的像函数一样**，你可以调用他们，将一个函数当做一个参数传到另外一个函数中等等。这是一个非常强大的特性让Python编程更加舒适甜美

允许一个类的实例像函数一样被调用。实质上说，这意味着 x() 与 x.\_\_call__() 是相同的。注意 \_\_call__ 参数可变。这意味着你可以定义 \_\_call__ 为其他你想要的函数，无论有多少个参数。

```python
class Entity:
    """调用实体来改变实体的位置。"""

    def __init__(self, size, x, y):
        self.x, self.y = x, y
        self.size = size

    def __call__(self, x, y):
        '''改变实体的位置'''
        self.x, self.y = x, y

    def print(self):
        print(self.x,self.y)

e = Entity(1, 2, 3) # 创建实例
e.print() # 2 3 
e(4, 5) #实例可以象函数那样执行，并传入x y值，修改对象的x y 
e.print() # 4 5 
```

一个有序的容器，神经网络模块将安按照在传入构造器的顺序依次被添加到计算图中执行，同时以神经网络模块为元素的有序字典也可以作为传入参数。
A sequential container.
Modules will be added to it in the order they are passed in the constructor.
Alternatively, an ordered dict of modules can also be passed in.

To make it easier to understand, here is a smale example:

```python
# Example of using Sequential
import torch
from torch import nn
from collections import OrderedDict

# Example of using Sequential
model1 = nn.Sequential(
            nn.Conv2d(1,20,5),
            nn.ReLU(),
            nn.Conv2d(20,64,5),
            nn.ReLU()
        )

print("model1: ",model1)
'''
model1:  Sequential(
  (0): Conv2d(1, 20, kernel_size=(5, 5), stride=(1, 1))
  (1): ReLU()
  (2): Conv2d(20, 64, kernel_size=(5, 5), stride=(1, 1))
  (3): ReLU()
)
'''


# Example of using Sequential with OrderedDict
model2 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(1,20,5)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv2d(20,64,5)),
            ('relu2', nn.ReLU())
        ]))
print("model2: ",model2)
'''
model2:  Sequential(
  (conv1): Conv2d(1, 20, kernel_size=(5, 5), stride=(1, 1))
  (relu1): ReLU()
  (conv2): Conv2d(20, 64, kernel_size=(5, 5), stride=(1, 1))
  (relu2): ReLU()
)
'''
            
```
有序字典 OrderedDict:
```python
import torch
from torch import nn 
import collections

d = {} # 无序字典
d['a'] = 'A'
d['b'] = 'B'
d['c'] = 'C'
for k in d:
    print(k) # a b c 这种遍历是输出 key

for k,v in d.items():
    print(k,v) # 这种遍历输出 key value
 
d1 = collections.OrderedDict() # 有序字典
d1['a'] = 'A'
d1['b'] = 'B'
d1['c'] = 'C'
d1['1'] = 1
d1['2'] = 2
for k,v in d1.items():
    print(k,v)

```

## 比较 torch.nn.Sequential 和 普通方法搭建的网络
```py
# 普通方法搭建的网络
class Net(nn.Module):
    def __init__(self,n_feature,n_hidden,n_output):
        super(Net,self).__init__()
        self.hidden = nn.Linear(n_feature,n_hidden)
        self.predict = nn.Linear(n_hidden,n_output)
    
    def forward(self,x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x 

model = Net(1,10,1)
print(model)
'''Net(
  (hidden): Linear(in_features=1, out_features=10, bias=True)
  (predict): Linear(in_features=10, out_features=1, bias=True)
)
这里没有激活层，激活层是在 前向推导中实现的，
'''
# 用Sequential搭建
model2 = nn.Sequential(
    nn.Linear(1,10),
    nn.ReLU(),
    nn.Linear(10,1)
)
print(model2)
'''
Sequential(
  (0): Linear(in_features=1, out_features=10, bias=True)
  (1): ReLU()
  (2): Linear(in_features=10, out_features=1, bias=True)
)
'''
```
区别：
> 使用torch.nn.Module，我们可以根据自己的需求改变传播过程，如RNN等
> 如果你需要快速构建或者不需要过多的过程，直接使用torch.nn.Sequential即可。