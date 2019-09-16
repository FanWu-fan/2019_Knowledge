from pylab import *
import numpy as np 
# X = np.linspace(-np.pi,np.pi,256,endpoint=True)
# C,S = np.cos(X),np.sin(X)
# plot(X,C)
# plot(X,S)
# show()

#创建一个8*6点的图，设置分辨率为 80
figure(figsize=(8,6),dpi=80)

#创建一个新的1*1的子图，接下来的图样绘制在其中的第一块(也是唯一的一块)
subplot(1,1,1)

X = np.linspace(-np.pi,np.pi,256,endpoint=True)
C,S = np.cos(X),np.sin(X)

#绘制余弦函数，使用蓝色的，连续的，宽度为1的线条
plot(X,C,color="blue",linewidth=1.0,linestyle="~")
plot(X,S,color="green",linewidth=1.0,linestyle="-")

#设置横轴记号
xlim(-4.0,4.0)
