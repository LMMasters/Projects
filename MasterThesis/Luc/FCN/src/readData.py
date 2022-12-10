import functools as ft
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
from src.yuFunctional import npf2i

# g(f(x)) -> F([x, f, g, ...])
F = lambda z: [*ft.reduce(lambda x, y: map(y, x), [z[:1]] + z[1:])][0]

# unpack data from file
def unpack(fname, homePath, ftail, usage = 1., signal = None):
    source = []
    files = [homePath + x for x in fname]
    for i, filename in enumerate(tqdm(files, leave = False)):
        print("File Name:", filename)
        hfr = h5py.File(filename, 'r')
        read = lambda x: F([x, hfr.get, np.array])
        temp_num = read("nobj")
        perm = F([temp_num.size, range, np.random.permutation])
        if type(usage) == int:
            temp_num = temp_num[perm][:usage]
        else:
            temp_num = temp_num[perm][:int(temp_num.size * usage)]
        temp_reg = read("reg")[perm][:temp_num.size]
        temp_cla = read("clas")[perm][:temp_num.size]
        if signal == None:
            temp_sig = np.array([i] * temp_num.size)
        print ("#objects", temp_num.dtype, temp_num.shape,
                temp_num.nbytes / 1E6, "MB")
        print ("4-momenta", temp_reg.dtype, temp_reg.shape,
                temp_reg.nbytes / 1E6, "MB")
        print ("classification", temp_cla.dtype, temp_cla.shape,
                temp_cla.nbytes / 1E6, "MB")
        if i:
            num = np.concatenate((num, temp_num))
            reg = np.concatenate((reg, temp_reg))
            cla = np.concatenate((cla, temp_cla))
            if signal == None:
                sig = np.concatenate((sig, temp_sig))
        else:
            num = temp_num
            reg = temp_reg
            cla = temp_cla
            if signal == None:
                sig = temp_sig
        source += [fname[i][:-ftail]] * temp_num.size
    if signal != None:
        sig = np.array([signal] * num.size)
    return num, reg, cla, sig, np.array(source)

# Test from Polina
def data_from_Polina(fname, homePath, ftail, usage = 1., permFlag = 1):
    source1, source2 = [], []
    files = [homePath + x for x in fname]
    for i, filename in enumerate(tqdm(files, leave = False)):
        print("File Name:", filename)
        hfr = h5py.File(filename, 'r')
        read = lambda x: F([x, hfr.get, np.array])
        temp_sig = read("type")
        if permFlag:
            perm = F([temp_sig.size, range, np.random.permutation])
        else:
            perm = F([temp_sig.size, range])
        temp_sig = temp_sig[perm]
        if type(usage) == int:
            temp_sig1 = temp_sig[:usage]
            temp_sig2 = temp_sig[usage:]
        else:
            temp_sig1 = temp_sig[:int(temp_sig.size * usage)]
            temp_sig2 = temp_sig[int(temp_sig.size * usage):]
        temp_num = read("nobj")[perm]
        temp_num1 = temp_num[:temp_sig1.size]
        temp_num2 = temp_num[temp_sig1.size:]
        temp_reg = read("reg")[perm]
        temp_reg1 = temp_reg[:temp_sig1.size]
        temp_reg2 = temp_reg[temp_sig1.size:]
        print ("#objects", temp_num1.dtype, temp_num1.shape,
                temp_num.nbytes / 1E6, "MB")
        print ("4-momenta", temp_reg1.dtype, temp_reg1.shape,
                temp_reg.nbytes / 1E6, "MB")
        print ("classification", temp_sig1.dtype, temp_sig1.shape,
                temp_sig.nbytes / 1E6, "MB")
        print ("#objects", temp_num2.dtype, temp_num2.shape,
                temp_num.nbytes / 1E6, "MB")
        print ("4-momenta", temp_reg2.dtype, temp_reg2.shape,
                temp_reg.nbytes / 1E6, "MB")
        print ("classification", temp_sig2.dtype, temp_sig2.shape,
                temp_sig.nbytes / 1E6, "MB")
        if i:
            num1 = np.concatenate((num1, temp_num1))
            reg1 = np.concatenate((reg1, temp_reg1))
            sig1 = np.concatenate((sig1, temp_sig1))
            num2 = np.concatenate((num2, temp_num2))
            reg2 = np.concatenate((reg2, temp_reg2))
            sig2 = np.concatenate((sig2, temp_sig2))
        else:
            num1 = temp_num1
            reg1 = temp_reg1
            sig1 = temp_sig1
            num2 = temp_num2
            reg2 = temp_reg2
            sig2 = temp_sig2
        source1 += [fname[i][:-ftail]] * temp_sig1.size
        source2 += [fname[i][:-ftail]] * temp_sig2.size
    return num1, reg1, sig1, np.array(source1),\
            num2, reg2, sig2, np.array(source2)

# Read XY from saved data
def read_XY_h5(name):
    hfr         = h5py.File(name, 'r')
    read        = [hfr.get, np.array]
    X_train     = F(["X1"] + read).astype("float32")
    X_test      = F(["X2"] + read).astype("float32")
    read        += [npf2i]
    Y_train     = F(["Y1"] + read).astype("int8")
    Y_test      = F(["Y2"] + read).astype("int8")
    data_cla    = F(['Z'] + read).astype("int8")
    return X_train, Y_train, X_test, Y_test, data_cla