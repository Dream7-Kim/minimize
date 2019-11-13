# %%
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all" 

%pylab inline
from pprint import pprint
from iminuit import Minuit


def line(x, a, b):
    return a + x * b

# data_x = linspace(0, 1, 10)
# # precomputed random numbers from a normal distribution
# offsets = array([-0.49783783, -0.33041722, -1.71800806,  1.60229399,  1.36682387,
#                  -1.15424221, -0.91425267, -0.03395604, -1.27611719, -0.7004073 ])
# data_y = line(data_x, 1, 2) + 0.1 * offsets # generate some data points with random offsets
# plot(data_x, data_y, "o")
# xlim(-0.1, 1.1)

def least_squares(a, b):
    yvar = 0.01
    return sum((data_y - line(data_x, a, b)) ** 2 / yvar)

m = Minuit(least_squares, a=5, b=5, fix_a=True,
           error_a=0.1, error_b=0.1,
           limit_a=(0, None), limit_b=(0, 10),
           errordef=1)
m.get_param_states()
m.migrad()

# %%
pprint(m.get_fmin())
# m.get_param_states()
# %%
pprint(m.fval)


# %%
m.hesse()


# %%
m.matrix(correlation=True)

# %%
