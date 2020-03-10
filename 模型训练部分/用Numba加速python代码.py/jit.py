from numba import jit

@njit    #or @jit(nopython=True)
def function(x):
    #your loop or numerically intensive computations
    return x
