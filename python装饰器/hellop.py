def decorator(func):
    def wrapper(*args,**kw):
        return func()
    return wrapper

@decorator
def function():
    print("hello,decorator")


function()