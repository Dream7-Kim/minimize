#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
np.random.random_sample((10000)) * 2.0 + 2.0


# In[3]:


import matplotlib
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1680)
## mass and width
M = 0.2
W = 0.01
## the k is constant of proportionality
g = np.sqrt(M * M * (M * M + W * W))
k = 2 * np.sqrt(2) * M * W * g / (np.pi * np.sqrt(M * M + g))
## example data
x = np.random.sample(1000) * 3 + 3  # data
mc = np.random.randn(100000) * 3 + 3  #MC
mc_bw = k / ((mc * mc - M * M)**2 + (M * W)**2)  # BW**2
mc_av = mc_bw.sum() / 100000
num_bins = 50

#fig, ax = plt.subplots()
## the histogram of the data
#n, bins, patches = ax.hist(x, num_bins, weights=np.exp(-(x-0.5)**2/0.04))
#n, bins, patches = ax.hist(x, num_bins, weights=1/((x-0.5)**2+0.01)/mc_av)

wt = k / ((x * x - M * M)**2 + (M * W)**2) / mc_av

def loglikelihood(c, w):
    mc = np.random.randn(100000) * 3 + 3
    mc_bw = k / ((mc * mc - c * c)**2 + (w * c)**2)
    mc_av = mc_bw.sum() / 100000
    _tmp = (np.log(k / (
        (x * x - c * c)**2 + (w * c)**2) * wt)).sum() - 1000 * np.log(mc_av)
    return _tmp


cc = np.arange(1000) / 1000
kk = []
for y in cc:
    kk.append(-loglikelihood(y, W))

plt.plot(cc, kk)

# add a 'best fit' line
#y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
#     np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
#ax.plot(bins, y, '--')
#ax.set_xlabel('Smarts')
#ax.set_ylabel('Probability density')
#ax.set_title(r'Histogram of IQ: $\mu=100$, $\sigma=15$')

# Tweak spacing to prevent clipping of ylabel
#fig.tight_layout()
plt.show()


# In[8]:


np.random.randn(437)


# In[ ]:





# In[14]:


N_points = 100000
n_bins = 20

# Generate a normal distribution, center at x=0 and y=5
x = np.random.randn(N_points)
y = .4 * x + np.random.randn(100000) + 5

fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

# We can set the number of bins with the `bins` kwarg
axs[0].hist(x, bins=n_bins)
axs[1].hist(y, bins=n_bins)

