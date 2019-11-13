#!/usr/bin/env python
# encoding: utf-8

import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import numpy as onp
import jax.numpy as np
import os
import jax
import time
import scipy.optimize as opt
import seaborn as sns
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

i = complex(
    0,
    1,
)


def read(string):
    lines = open(string).readlines()
    lists = []
    num = 0
    for line in lines:
        str = line.replace('[', '')
        str = str.replace(']', '')
        str = str.strip()
        tmp = float(str)
        lists.append(tmp)
        num = num + 1
        if (num == 50000 * 4):
            break
    array = onp.array(lists).reshape(int(num / 4), 4)
    return array


def readMC(string):
    lines = open(string).readlines()
    lists = []
    num = 0
    for line in lines:
        str = line.replace('[', '')
        str = str.replace(']', '')
        str = str.strip()
        tmp = float(str)
        lists.append(tmp)
        num = num + 1
        if (num == 700000 * 4):
            break
    array = onp.array(lists).reshape(int(num / 4), 4)
    return array


def BW(mass, width, Pb, Pc):
    Pbc = Pb + Pc
    _Pbc = Pbc * np.array([-1, -1, -1, 1])
    Sbc = onp.sum(Pbc * _Pbc, axis=1)
    k = np.sqrt(width)
    return k / (mass**2 - Sbc - i * mass * width)


def phase(theta, rho):
    return rho * np.exp(theta*i)


phif001 = (read('data/phif001MC.txt'))[:,0:2]
Kp = read('data/KpMC.txt')
Km = read('data/KmMC.txt')
Pip = read('data/PipMC.txt')
Pim = read('data/PimMC.txt')

phif001MC = (readMC('data/phif001MC.txt'))[:,0:2]
KpMC = readMC('data/KpMC.txt')
KmMC = readMC('data/KmMC.txt')
PipMC = readMC('data/PipMC.txt')
PimMC = readMC('data/PimMC.txt')


def modelf0(var, phif001, Kp, Km, Pip, Pim):
    up_phif001 = phif001.T * \
        BW(var[0], var[1], Kp, Km) * BW(var[2], var[3], Pip, Pim)
    up_1 = up_phif001
    conj_up_1 = np.conj(up_1)
    up_2 = np.real(np.sum(up_1 * conj_up_1, axis=0))/2
    return up_2

var_weight = np.array([1.02, 0.004, 1.37, 0.35])
weight_ = modelf0(var_weight, phif001, Kp, Km, Pip, Pim)


def model(var):
    up_phif001 = phif001.T * \
        BW(var[0], var[1], Kp, Km) * BW(var[2], var[3], Pip, Pim)
    up_1 = up_phif001
    conj_up_1 = np.conj(up_1)
    up_2 = np.real(np.sum(up_1 * conj_up_1, axis=0))/2
    low_phif001 = phif001MC.T * \
        BW(var[0], var[1], KpMC, KmMC) * BW(var[2], var[3], PipMC, PimMC)
    low_1 = low_phif001
    conj_low_1 = np.conj(low_1)
    low_2 = np.real(np.sum(low_1 * conj_low_1, axis=0))/2
    low_3 = np.average(low_2)
    dim = (up_2.shape)[0]
    return -(np.sum(np.log(up_2) * weight) - dim * np.log(low_3))





# cc = np.arange(1000) / 100
# kk = []
# for y in cc:
#     _var = onp.array([1.02, 0.001, y, 0.35])
#     kk.append(model(_var, phif001, phif001MC, Kp, Km,
#                     Pip, Pim, KpMC, KmMC, PipMC, PimMC, weight_))

# plt.plot(cc, kk)
# plt.savefig('new.png')
# # scipy optimize

grad = jax.jit(jax.grad(model))

# bounds = [(1.01, 1.04), (0, 0.005), (1.3, 1.4), (0.3, 0.4)]
# minimizer_kwargs = {"method":"BFGS", "jac":grad}
# result = opt.shgo(model, bounds, n=1000, sampling_method='sobol', minimizer_kwargs=minimizer_kwargs, options={"disp":True}, args=(phif001, phif001MC, Kp, Km, Pip, Pim, KpMC, KmMC, PipMC, PimMC, weight_))

x0 = [1.02, 0.004, 1.37, 0.35]
print('------------------ BFGS ------------------')
res = opt.minimize(model, x0, method='BFGS', jac=grad, options={'disp': True})
print('opt res:', res.x)
print('opt res Hessian:', res.hess_inv)
print('\n\n\n')