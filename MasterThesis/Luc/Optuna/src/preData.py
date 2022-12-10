import functools as ft
import numpy as np
from src.yuFunctional import npf2i

# g(f(x)) -> F([x, f, g, ...])
F = lambda z: [*ft.reduce(lambda x, y: map(y, x), [z[:1]] + z[1:])][0]

# Normalization
def normalize(data):
    mean = np.mean(data, axis = 0)
    std = np.std(data, axis = 0)
    data = (data - mean) / np.max(([1] * std.size, std), axis = 0)
    return data, mean, std


# Encode 4 momenta
def encode(data, vec):
    head = data[:2]
    body = data[2:50].reshape(-1, 4)
    tail = data[50:].reshape(-1, 1)
    body = np.concatenate((body, tail), 1)
    expand = lambda x: np.outer(x[:-1], vec[npf2i(x[-1]) - 1]).T.flatten()
    tail = np.apply_along_axis(expand, 1, body).flatten()
    return np.concatenate((head, tail))

def permutation(*data):
    perm = F([data[0].shape[0], range, np.random.permutation])
    return [x[perm] for x in data]