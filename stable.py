# %%
import matplotlib.pyplot as plt
import numpy as onp
import jax.numpy as np
import os
import jax
import time
import scipy.optimize as opt
import seaborn as sns
import iminuit
from pprint import pprint

M1 = 1.06
W1 = 0.03
M2 = 1.37
W2 = 0.1
M3 = 2.91
W3 = 0.2
rho1 = 0.6
M4 = 3.52
W4 = 0.15
N = 20000

# 如果两个峰间距太近会混在一起，width

data0 = onp.random.sample(N) * 4
data1 = onp.random.sample(N) * 4


i = complex(0, 1)


def BW(Sbc, m_, w_):
    gamma = np.sqrt(m_*m_*(m_*m_+w_*w_))
    k = np.sqrt(2*np.sqrt(2)*m_*np.abs(w_)*gamma/np.pi/np.sqrt(m_*m_+gamma))
    return k/(m_*m_ - Sbc**2 - i*m_*w_)


def WT(bw_):
    conjbw = np.conj(bw_)
    return np.real(bw_*conjbw)


# wt1 = WT(BW(data0,M1,W1))
wt1 = 0
wt2 = WT(BW(data1, M2, W2)*rho1 + BW(data1, M3, W3) + BW(data1, M4, W4))


def likelihood(var):
    # b1 = BW(data0, var[0], var[1])
    b2 = BW(data1, var[0], var[1])*rho1 + BW(data1, var[2], var[3])  + BW(data1, var[4], var[5])
    # return -(np.log(WT(b1))*wt1 + np.log(WT(b2))*wt2).sum()
    return -(np.log(WT(b2))*wt2).sum()


# cc = np.arange(1000) / 100
# kk = []
# for y in cc:
#     var = np.array([M1, W1, M2, y, M3, W3])
#     kk.append(likelihood(var))
# plt.plot(cc, kk)
# plt.grid()
# plt.savefig('mutil_rho.png')

print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
grad = jax.jit(jax.grad(likelihood))
x = np.array([1.4, 0.11, 2.91, 0.01, 3.5, 0.02])
res = opt.minimize(likelihood, x, method='BFGS',
                   jac=grad, options={'disp': True})
np.set_printoptions(precision=16)
print('\n\nResult: ', res.x)
print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')


m = iminuit.Minuit.from_array_func(likelihood, x, grad=grad, error=0.001, errordef=0.5)
print('\n--------------------------Initial state')
print(m.get_initial_param_states())

m.migrad()

print('\n--------------------------After migrad')
print(m.get_param_states())
print('\n--------------------------hesse')
print(m.hesse())
# pprint(m.hesse())
# print(type(m.hesse()))
print('\n--------------------------last_f')
res = m.get_fmin()
print(res)
print(res.fval)
print(res.is_valid)
# print(type(m.get_fmin()))

m.minos()
print('\n--------------------------After minos')
print(m.get_param_states())
# print(type(m.get_param_states()))

# %%
param = m.get_param_states()
merror = m.get_merrors()
for p in param:
    print(p)
    print(merror[p.name])

