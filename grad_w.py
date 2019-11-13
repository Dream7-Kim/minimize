import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want
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

M1 = 1.02
W1 = 0.004
M2 = 1.37
W2 = 0.35


def BW(mass, width, Pb, Pc):
    r = np.sqrt(mass * mass * (mass * mass + width * width))
    k = (2 * np.sqrt(2) * mass * width * r / np.pi / np.sqrt(mass * mass + r))
    Pbc = Pb + Pc
    _Pbc = Pbc * np.array([-1, -1, -1, 1])
    Sbc = np.sum(Pbc * _Pbc, axis=1)
    result = k / ((mass**2 - Sbc)**2 + (mass * width)**2)
    return result


def phase(theta, rho):
    return rho * np.exp(theta * i)


def modelf0(W):
    up_phif001 = BW(M1, W, Kp, Km) * BW(M2, W2, Pip, Pim)
    # index = np.where(up_phif001)
    return up_phif001


weight = modelf0(W1)


def model(W):
    up_phif001 = BW(M1, W, Kp, Km) * BW(M2, W2, Pip, Pim)
    up = up_phif001
    low_phif001 = BW(M1, W, KpMC, KmMC) * BW(M2, W2, PipMC, PimMC)
    low = np.average(low_phif001)
    dim = (up.shape)[0]
    result = np.sum(np.log(up) * weight) - dim * np.log(low)
    return -result


grad = jax.jit(jax.grad(model))
# hess =  jax.jit(jax.hessian(model))

# x_var = onp.arange(0, 0.1, 0.001)
# y_var = []
# i = 0
# for x in x_var:
#     i = i + 1
#     # print('Step', i)
#     if(i % 100 == 0):
#         print('Step', i)
#     y_var.append(model(x))

# fig = plt.figure(figsize=(12, 12))
# plt.plot(x_var, y_var)
# plt.grid()
# plt.savefig('one_var.png')

# y_grad = []
# i = 0
# for x in x_var:
#     i = i + 1
#     # print('Step', i)
#     if(i % 100 == 0):
#         print('Step', i)
#     y_grad.append(grad(x))

# fig = plt.figure(figsize=(12, 12))
# plt.plot(x_var, y_grad)
# plt.grid()
# plt.savefig('one_var_grad.png')

x0 = [0.0042]
# print('------------------ BFGS ------------------')
# res = opt.minimize(model, x0, method='BFGS', jac=grad, options={'disp': True})
# print('opt res:', res.x)
# print('opt res Hessian:', res.hess_inv)
# print('\n\n\n')
res = opt.minimize(model, x0, method='dogleg', jac=grad, hess=hess, options={'xtol': 1e-8, 'disp': True})
print(res.x)

