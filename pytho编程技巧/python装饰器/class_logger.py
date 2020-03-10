class logger(object):
    def __init__(self,func):
        self.func = func
    
    def __call__(self,*args,**kwargs):
        print(f"[INFO]: the function {self.func.__name__} is running")
        return self.func(*args,**kwargs)

@logger
def say(something):
    print(f"say {something}!")

say("hello")