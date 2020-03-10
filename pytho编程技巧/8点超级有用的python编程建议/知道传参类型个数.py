def add(a, b):
    if isinstance(a,int) and isinstance(b,int):
        return a+b
    else:
        return '参数类型错误'
print(add(1,2))
print(add(1,'a'))