
# 这是装饰函数
def timer(func):
    def wrapper(*args,**kw):
        t1 = time.time()
        # 这是函数真正执行的地方
        func(*args,**kw)

        t2 = time.time()

        cost_time = t2 -t1 
        print(f"花费时间：{cost_time}")
    return wrapper

import time
@timer
def want_sleep(sleep_time):
    time.sleep(sleep_time)

want_sleep(10)











