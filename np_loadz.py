import numpy as np

def np_loadz(fn):
    data = np.load(fn)
    keys = [x for x in data]
    return data[keys[0]]