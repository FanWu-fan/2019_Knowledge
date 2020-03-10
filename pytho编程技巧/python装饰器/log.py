# 这是装饰器函数，参数 func是被装饰的函数

def logger(func):
    def wrapper(*args,**kw):
        print(f"我要开始执行：{func.__name__} 函数了")
    
        #真正执行的是这行
        func(*args,**kw)

        print('执行完毕')
    return wrapper

@logger
def add(x,y):
    print(f"{x} + {y} = {x+y}")

add(200,50)