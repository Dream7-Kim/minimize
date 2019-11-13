# %%
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
import iminuit

# %pylab inline
from pprint import pprint

M1 = 1.06
W1 = 0.03
M2 = 1.37
W2 = 0.1
# M3 = [1.58, 1.75, 1.91, 2.32, 2.64, 2.84, 3.12, 3.54, 3.79]
M3 = [1.38, 1.75]
W3 = []
for i in range(len(M3) - 1):
    W3.append((M3[i+1] - M3[i]) / 5)
W3.append(0.03)

M3 = []
W3 = []

N = 20000
const = onp.random.sample(len(M3)) * 10
# print(const)
# print(M3)
# print(W3)

# 如果两个峰间距太近会混在一起，width

data0 = onp.random.sample(N) * 4
data1 = onp.random.sample(N) * 4
data2 = onp.random.sample(N) * 4


i = complex(0,1)

def BW(Sbc,m_,w_):
    gamma=np.sqrt(m_*m_*(m_*m_+w_*w_))
    k = np.sqrt(2*np.sqrt(2)*m_*np.abs(w_)*gamma/np.pi/np.sqrt(m_*m_+gamma))
    return k/(m_*m_ - Sbc**2 - i*m_*w_)

def WT(bw_):
    conjbw = np.conj(bw_)
    return np.real(bw_*conjbw)


wt1 = WT(BW(data0,M1,W1))
temp = BW(data1, M2, W2)
for i in range(len(M3)):
    temp = temp + const[i] * BW(data1, M3[i], W3[i]) 
wt2 = WT(temp)

def likelihood(var):
    b1 = BW(data0, var[0], var[1])
    b2 = BW(data1, var[2], var[3])
    # print(var)
    for i in range(len(M3)):
        # print(i, 4 + 2*i, var[4 + 2*i])
        b2 = b2 + BW(data1, var[4 + 2*i], var[5 + 2*i])
    return -(np.log(WT(b1))*wt1 + np.log(WT(b2))*wt2).sum()


# cc = np.arange(1000) / 2000  + 0.1
# kk = []
# for y in cc:
#     var = np.array([M1,W1,M2,W2,M3,y])
#     kk.append(likelihood(var))
# plt.plot(cc,kk)
# plt.grid()
# plt.savefig('mutil.png')


num_var = 4 + 2*len(M3)
# print('Argument numbers:', num_var)

grad = jax.jit(jax.grad(likelihood))
x_ = [1.04, 0.05, 1.3, 0.44]
for i in range(len(M3)):
    x_.append(M3[i] - 0.1)
    x_.append(onp.abs(W3[i] - 0.01))

x = onp.array(x_)

# pprint(likelihood(x))
# pprint(grad(x))

# %%
error = onp.zeros(num_var)
for i in range(num_var):
    error[i] = 0.001

name = []
fixed = []
for i in range(num_var):
    fixed.append(True)
    if(i%2==0):
        name.append('M_like_'+str(i/2 + 1))
    else:
        name.append('W_like_'+str((i-1)/2 + 1))

fixed[0] = False
fixed[1] = False
# fixed[2] = False
# fixed[3] = False

pprint(iminuit.minimize(likelihood, x_))


# %%
m = iminuit.Minuit.from_array_func(likelihood, x_, grad=grad, error=error, fix=fixed, name=name, errordef=0.5)
print('\n--------------------------Initial state')
print(m.get_initial_param_states())

m.migrad()
print('\n--------------------------After migrad')
print(m.get_param_states())
print('\n--------------------------hesse')
pprint(m.hesse())
print('\n--------------------------last_f')
pprint(m.get_fmin())

m.minos()
print('\n--------------------------After minos')
print(m.get_param_states())


# %%
# x_store = x
# y = []
# for xx in onp.arange(0.9, 1.3, 0.01):
#     temp = x_store
#     temp[0] = xx
#     # print(temp)
#     y.append(likelihood(temp))

# plt.plot(onp.arange(0.9, 1.3, 0.01), y)
# plt.savefig('test.png')