# %%
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
    row = int(len(lines)/4)
    lists = []
    for line in lines:
        str = line.replace('[', '')
        str = str.replace(']', '')
        str = str.strip()
        tmp = float(str)
        lists.append(tmp)
    array = onp.array(lists).reshape(row, 4)
    return array


def BW(mass, width, Pb, Pc):
    Pbc = Pb + Pc
    _Pbc = Pbc * np.array([-1, -1, -1, 1])
    Sbc = onp.sum(Pbc * _Pbc, axis=1)
    return 1 / (mass**2 - Sbc - i * mass * width)


def phase(theta, rho):
    return rho * np.exp(theta*i)


phif001 = read('data/phif001MC.txt')
phif021 = read('data/phif021MC.txt')
Kp = read('data/KpMC.txt')
Km = read('data/KmMC.txt')
Pip = read('data/PipMC.txt')
Pim = read('data/PimMC.txt')


def modelf0(var, phif001, phif001MC, phif021, phif021MC, Kp, Km, Pip, Pim, KpMC, KmMC, PipMC, PimMC):
    up_phif001 = phif001.T * BW(var[0], var[1], Kp,
                                Km) * BW(var[2], var[3], Pip, Pim)
    up_phif021 = phif021.T * BW(var[0], var[1], Kp,
                                Km) * BW(var[2], var[3], Pip, Pim)
    # print(up_phif001.shape)
    up_1 = (up_phif001 + up_phif021)
    # print(up_1.shape)
    up_2 = np.vstack([up_1[0, :], up_1[1, :]])
    # print(up_2.shape)
    conj_up_2 = np.conj(up_2)
    up_3 = np.real(np.sum(up_2 * conj_up_2, axis=0))/2
    # print(up_3.shape)
    low_phif001 = phif001MC.T * \
        BW(var[0], var[1], KpMC, KmMC) * BW(var[2], var[3], PipMC, PimMC)
    low_phif021 = phif021MC.T * \
        BW(var[0], var[1], KpMC, KmMC) * BW(var[2], var[3], PipMC, PimMC)
    low_1 = (low_phif001 + low_phif021)
    low_2 = np.vstack([low_1[0, :], low_1[1, :]])
    conj_low_2 = np.conj(low_2)
    low_3 = np.real(np.sum(low_2 * conj_low_2, axis=0))/2
    # print(low_3.shape)
    dim = (low_3.shape)[0]
    # print(dim)
    low_4 = np.sum(low_3)/dim
    return up_3 / low_4


def model(var, phif001, phif001MC, phif021, phif021MC, Kp, Km, Pip, Pim, KpMC, KmMC, PipMC, PimMC, weight):
    up_phif001 = phif001.T * BW(var[0], var[1], Kp,
                                Km) * BW(var[2], var[3], Pip, Pim)
    up_phif021 = phif021.T * BW(var[0], var[1], Kp,
                                Km) * BW(var[2], var[3], Pip, Pim)
    # print(up_phif001.shape)
    up_1 = (up_phif001 + up_phif021)
    # print(up_1.shape)
    up_2 = np.vstack([up_1[0, :], up_1[1, :]])
    # print(up_2.shape)
    conj_up_2 = np.conj(up_2)
    up_3 = np.real(np.sum(up_2 * conj_up_2, axis=0))/2
    # print(up_3.shape)
    low_phif001 = phif001MC.T * \
        BW(var[0], var[1], KpMC, KmMC) * BW(var[2], var[3], PipMC, PimMC)
    low_phif021 = phif021MC.T * \
        BW(var[0], var[1], KpMC, KmMC) * BW(var[2], var[3], PipMC, PimMC)
    low_1 = (low_phif001 + low_phif021)
    low_2 = np.vstack([low_1[0, :], low_1[1, :]])
    conj_low_2 = np.conj(low_2)
    low_3 = np.real(np.sum(low_2 * conj_low_2, axis=0))/2
    # print(low_3.shape)
    dim = (low_3.shape)[0]
    # print(dim)
    low_4 = np.sum(low_3)/dim
    return -np.sum(np.log(up_3/low_4)*weight)


var_weight = np.array([1.02, 0.004, 1.37, 0.35])
weight_ = modelf0(var_weight, phif001[:50000], phif001[50000:5000000], phif021[:50000], phif021[50000:5000000], Kp[:50000],
                  Km[:50000], Pip[:50000], Pim[:50000], Kp[50000:5000000], Km[50000:5000000], Pip[50000:5000000], Pim[50000:5000000])
# scipy optimize
grad = jax.jit(jax.grad(model))
var_ = np.array([1.02, 0.004, 1.35, 0.37])
result = model(var_, phif001[:50000], phif001[50000:500000], phif021[:50000], phif021[50000:500000], Kp[:50000],
               Km[:50000], Pip[:50000], Pim[:50000], Kp[50000:500000], Km[50000:500000], Pip[50000:500000], Pim[50000:500000], weight_)
# print(result)
args_ = (phif001[:50000], phif001[50000:500000], phif021[:50000],
         phif021[50000:500000],
         Kp[:50000], Km[:50000], Pip[:50000], Pim[:50000], Kp[50000:500000], Km[50000:500000], Pip[50000:500000], Pim[50000:500000], weight_)
start = time.time()
# print('*************** Global Optimization ***************')
# bounds = [(-1000, 1000), (-1000, 1000), (-1000, 1000), (-1000, 1000)]
bounds = [(1.01, 1.04), (0, 0.005), (1.3, 1.4), (0.3, 0.4)]
minimizer_kwargs = {"method":"BFGS", "jac":grad}
result = opt.shgo(model, bounds, n=10, sampling_method='sobol', minimizer_kwargs=minimizer_kwargs, options={"disp":True}, args=args_)
# result = opt.shgo(fun, bounds, n=100, sampling_method='sobol', minimizer_kwargs=minimizer_kwargs, options={"disp":False})

# %%
