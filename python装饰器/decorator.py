

def say_hello(contry):
    def wrapper(func):
        def deco(*args,**kwargs):
            if contry == "china":
                print("你好呀")
            elif contry == "america":
                print("hello")
            else:
                return

            # 真正执行函数的地方
            func(*args,**kwargs)
        return deco
    return wrapper

# 小明，中国人
@say_hello("china")
def xiaoming():
    pass

# jack 美国人
@say_hello("america")
def jack():
    pass

xiaoming()
print("---------")
jack()