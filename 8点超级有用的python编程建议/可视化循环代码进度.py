from tqdm import tnrange,tqdm_notebook
from time import sleep

for i in tqdm_notebook(range(10),desc = 'lst loop'):
    for j in tnrange(100,desc = '2nd loop',leave = False):
        sleep(0.01)