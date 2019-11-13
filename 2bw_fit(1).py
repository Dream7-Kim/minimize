#!/usr/bin/env python
# coding: utf-8

# In[15]:


import matplotlib.pyplot as plt
import numpy as onp
import jax.numpy as np
import os
import jax
import time
import scipy.optimize as opt
import seaborn as sns
from matplotlib import cm

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
    Sbc = np.sum(Pbc * _Pbc, axis=1)
    result = width / ((mass**2 - Sbc)**2 + (mass * width)**2)
    return result


# def BW(mass, width, Sbc):
#     result = 1 / ((mass**2 - Sbc**2)**2 + (mass * width)**2)
#     return result


def phase(theta, rho):
    return rho * np.exp(theta * i)


def modelf0(var, Kp, Km, Pip, Pim):
    up_phif001 = BW(var[0], var[1], Kp, Km) * BW(var[2], var[3], Pip, Pim)
    return up_phif001


def model(var, Kp, Km, KpMC, KmMC, Pip, Pim, PipMC, PimMC, weight):
    up_phif001 = BW(var[0], var[1], Kp, Km) * BW(var[2], var[3], Pip, Pim)
    up = up_phif001
    low_phif001 = BW(var[0], var[1], KpMC, KmMC) * BW(var[2], var[3], PipMC,
                                                      PimMC)
    low = np.average(low_phif001)
    dim = (up.shape)[0]
    result = np.sum(np.log(up) * weight) - dim * np.log(low)
    return -result


M = 1.02
W = 0.004

phif001 = read('data/phif001MC.txt')
Kp = read('data/KpMC.txt')
Km = read('data/KmMC.txt')
Pip = read('data/PipMC.txt')
Pim = read('data/PimMC.txt')

phif001MC = readMC('data/phif001MC.txt')
KpMC = readMC('data/KpMC.txt')
KmMC = readMC('data/KmMC.txt')
PipMC = readMC('data/PipMC.txt')
PimMC = readMC('data/PimMC.txt')

var_ = onp.array([M, W, 1.37, 0.35])
# Sbc = onp.random.randn(10000) / 100 + 1.015
# SbcMC = onp.random.randn(100000) / 100 + 1.015
# weight1 = modelf0(var_, Pip, Pim, PipMC[50000:600000], PimMC[50000:600000])
weight1 = modelf0(var_, Kp, Km, Pip, Pim)
# weight1 = modelf0(var_, Sbc)

# result = model(var_, Pip, Pim, PipMC[50000:600000], PimMC[50000:600000],
#                weight1)

cc = np.arange(1000) / 10000 - 0.01
kk = []
for y in cc:
    _var = onp.array([0.01207, y, 1.37, 0.35])   
    kk.append(model(_var,Kp,Km,KpMC,KmMC,Pip,Pim,PipMC,PimMC,weight1))
#         model(_var, Sbc, SbcMC, weight1))

plt.plot(cc, kk)


# grad = jax.jit(jax.grad(model))

x0 = [1.022, 0.0042, 1.372, 0.352]
print('------------------ BFGS ------------------')
res = opt.minimize(model, x0, method='BFGS', jac=grad, options={'disp': True}, args=(Kp,Km,KpMC,KmMC,Pip,Pim,PipMC,PimMC,weight1))
print('opt res:', res.x)
print('opt res Hessian:', res.hess_inv)
print('\n\n\n')






# %%
