

```python
import torch
from torch import nn 
import torch.nn.functional as F 
import torchvision
```

创建未初始化的 **Tensor**,并不代表为0，只是代表随机值  empty


```python
x = torch.empty(5,3)
print(x,type(x))
```

    tensor([[-3.4980e-13,  7.4969e-43, -3.4980e-13],
            [ 7.4969e-43, -3.4980e-13,  7.4969e-43],
            [-3.4980e-13,  7.4969e-43, -3.4980e-13],
            [ 7.4969e-43, -3.4980e-13,  7.4969e-43],
            [-3.4980e-13,  7.4969e-43, -3.4980e-13]]) <class 'torch.Tensor'>
    

创建一个随机初始化的 Tensor,  rand


```python
x = torch.rand(5,3)
print(x)
```

    tensor([[0.4848, 0.1496, 0.0987],
            [0.3356, 0.5657, 0.9435],
            [0.0467, 0.6408, 0.0962],
            [0.6664, 0.2442, 0.7227],
            [0.7755, 0.2541, 0.1663]])
    

创建全零long型Tensor zeros


```python
x = torch.zeros(5,3,dtype=torch.long)
print(x)
```

    tensor([[0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]])
    


```python
x = torch.linspace(1,10,5)
print(x)
```

    tensor([ 1.0000,  3.2500,  5.5000,  7.7500, 10.0000])
    

### 索引

我们可以使用类似 Numpy的索引操作来访问 Tensor的一部分，需要注意的是： **索引出来的结果与原数据共享内存，也即是修改一个，另外一个也会跟着修改**


```python
y = x[0: ]
print(y)
y +=1
print(y)
print(x)
```

    tensor([ 2.0000,  4.2500,  6.5000,  8.7500, 11.0000])
    tensor([ 3.0000,  5.2500,  7.5000,  9.7500, 12.0000])
    tensor([ 3.0000,  5.2500,  7.5000,  9.7500, 12.0000])
    


```python
torch.nonzero(y)
```




    tensor([[0],
            [1],
            [2],
            [3],
            [4]])



### 改变形状

view() 返回的新的 Tensor与 源 Tensor虽然可能有不同的 size，但是共享 data的，也即是更改其中的一个，另外一个也会跟着改变。(顾名思义，view仅仅是改变了对这个张量的观察角度，内部数据并未改变)。 虽然view返回的 Tensor和 源Tensor是共享 data的，但是仍然是一个新的 Tensor（因为Tensor 除了包含 data外还有一些其他属性），两者的id（内存地址）并不一致。

如果想返回一个真正新的副本(即不共享data内存)该怎么办？推荐使用 clone创造一个副本然后再使用 view（）



```python
x = torch.randn(3,5)
x_cp = x.clone().view(15)
print(x)
print(x_cp)
```

    tensor([[-0.5193, -0.7769, -1.2784,  0.0956,  0.3346],
            [-0.2425, -0.8835,  0.2302,  1.5465, -0.9690],
            [ 1.3337, -1.3048,  0.3819, -0.1265, -0.9160]])
    tensor([-0.5193, -0.7769, -1.2784,  0.0956,  0.3346, -0.2425, -0.8835,  0.2302,
             1.5465, -0.9690,  1.3337, -1.3048,  0.3819, -0.1265, -0.9160])
    


```python
x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
id_before = id(y)
y = y + x
print(id(y) == id_before) # False 

```

    False
    


```python
x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
id_before = id(y)
y[:] = y + x
print(id(y) == id_before) # True

```

    True
    

### Tensor 和 Numpy 相互转换

我们很容易用 numpy() 和 from_numpy() 将 Tensor 和 Numpy中的数组相互转化。但是需要注意的一点是：**这两个函数产生的 Tensor 和 Numpy中的数组共享相同的内存（所以他们之间转换很快），改变其中一个时另外一个也会改变**。 还有一个常用的将numpy中的 array转换成 tensor 的方法是 torch.tensor,这种方法会进行数据拷贝，返回的 tensor和原来的数据不再共享内存


```python
a = torch.ones(5)
b = a.numpy()
print(a, b)

a += 1
print(a, b)
b += 1
print(a, b)

```

    tensor([1., 1., 1., 1., 1.]) [1. 1. 1. 1. 1.]
    tensor([2., 2., 2., 2., 2.]) [2. 2. 2. 2. 2.]
    tensor([3., 3., 3., 3., 3.]) [3. 3. 3. 3. 3.]
    


```python
a = torch.ones(5)
b= a.numpy()
print(id(a.data),id(b.data))
print(a,b)
a += 1
print(a.data)
print(b.data)
```

    2300710437152 2300713126600
    tensor([1., 1., 1., 1., 1.]) [1. 1. 1. 1. 1.]
    tensor([2., 2., 2., 2., 2.])
    <memory at 0x00000217AD304AC8>
    


```python
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
print(a, b)

a += 1
print(a, b)
b += 1
print(a, b)



```

    [1. 1. 1. 1. 1.] tensor([1., 1., 1., 1., 1.], dtype=torch.float64)
    [2. 2. 2. 2. 2.] tensor([2., 2., 2., 2., 2.], dtype=torch.float64)
    [3. 3. 3. 3. 3.] tensor([3., 3., 3., 3., 3.], dtype=torch.float64)
    


```python
c = torch.tensor(a)
a += 1
print(a, c)

```

    [4. 4. 4. 4. 4.] tensor([3., 3., 3., 3., 3.], dtype=torch.float64)
    


```python

```
