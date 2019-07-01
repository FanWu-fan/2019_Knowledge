import time 
tic = time.time()
much_job = [x**2 for x in range(1,1000000,3)]
toc = time.time()
print('used {:.5}s'.format(toc-tic))
# 0.11367s