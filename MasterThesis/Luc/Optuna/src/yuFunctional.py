import functools as ft
import numpy as np

## Functions and Functionals 

# np.round() return int instead of .0
def npf2i(x):
    x = np.array(x).round().astype("int64")
    if x.size == 1:
        return int(x)
    else:
        return x