import numpy as np
from scipy.optimize import minimize
import scipy.optimize as optimize

from sympy.utilities.lambdify import lambdify
# from sympy.mpmath import *
import sympy as sy
from sympy.matrices import hessian
from sympy.physics.vector import gradient
from sympy import Matrix
from sympy.solvers import solve
from sympy import Symbol
from sympy import *
import time

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.ticker as ticker


# import scipy as sp
# import matplotlib as mpl
# from sympy import Function
##from sympy.abc import x #from http://docs.sympy.org/dev/modules/numeric-computation.html 
# from sympy.utilities.lambdify import lambdify
# import sympy as sy

################http://stackoverflow.com/questions/31865549/want-to-do-multi-variation-minimize-with-sympy####
# sy.init_printing()  # LaTeX like pretty printing for IPython


def scalar_masses(f, fields):
    h = hessian(f, fields)
    (p, d) = h.diagonalize()
    return np.diagonal(d)


def solve_one_loop(f, fields, masses, vevs, unknown_constants):
    vevtuple = solve(np.subtract(fields, vevs))

    firstder = ([sy.diff(f, i).subs(vevtuple) for i in fields])
    # may have to modify in the situation where a field has only mass term in pot
    diaghess = ([m.subs(vevtuple) for m in scalar_masses(f, fields)])  # subs only work on syMatrices

    lhs = tuple(firstder) + tuple(diaghess)
    rhs = tuple([0. for i in fields]) + tuple([m ** 2 for m in masses])
    return solve(list(np.subtract(lhs, rhs)), unknown_constants, check=True)  # ,set=True


def f_v(zz):
    """ Helper for receiving vector parameters """
    return f_n(zz[0], zz[1], zz[2], zz[3], zz[4])


def jac_v(zz):
    """ Jacobian Helper for receiving vector parameters """
    return np.array([jfn(zz[0], zz[1], zz[2], zz[3], zz[4]) for jfn in jac_fn])


def minimise(bounds, initial):
    # optimisation
    res = minimize(f_v, initial, method='SLSQP', jac=jac_v, bounds=bounds, tol=1e-12)

    # other ways to optimise
    # res=minimize(f_v,initial,method='Nelder-Mead')
    # rslts = minimize(f_v,zz0, method='BFGS', jac=jac_v)
    # rslts = optimize.fmin(f_v,bnds)
    # rslts = minimize(f_v,zz0)
    # optimize.fmin_cg(f_v, [-10., 10.])

    # output results
    print(("field-constant values = " + len(res.x) * "%8.6f\t\t") % tuple(res.x))
    print("potential value = %8.4f" % res.fun)
    return res.x


###############################################################################
# TO PUT IN SEPARATE FILE
###############################################################################

# define fields and constants
x1, x2, c20, c30, c40 = sy.symbols('x1 x2  c20 c30 c40')
fields = (x1, x2)
constants = (c20, c30, c40)
xx = fields + constants

# potential currently cannot handle pure mass terms
f = c20 * x1 ** 2 + c30 * x1 ** 3 + c40 * x1 ** 4  # + x2**4

# sy.Matrix((f,f2)).subs({x1:246.,x2:0})
# define the vev and masses to solve for the 1st derivative and diagonalised hessian
# (field dependent masses) of the potential
vevs = (246., 0.)
masses = (0., 125.)  # some ordering may be required to match diagonalisation

# Build Jacobian if require to be used in function "minimise:
jac_f = [f.diff(x) for x in xx]
jac_fn = [lambdify(xx, jf, modules='numpy') for jf in jac_f]

# lambdify function for faster handling
# ######lambify with arbitrary arguments
# http://stackoverflow.com/questions/34115233/python-optimization-using-sympy-lambdify-and-scipy
f_n = lambdify(xx, f, modules='numpy')

# solve the constants from 1st derivative and hessian of potential
sol = solve_one_loop(f, fields, masses, vevs, (c20, c40))  # specify redundent param in last arguement

print("=======printing solutions to solving constants========")
print(sol)
print("======================================================")

# ####working out the free-parameters######
f_reduced = f.subs(sol)
remain_var = tuple(f_reduced.free_symbols - set(fields))
print(remain_var)

# currently just setting values for the constants but should use the values from "solveoneloop"
c20val = c20.subs(sol).subs({c30: -1.5 * 125. ** 2 / 246. / 3})
c30val = c30.subs(sol).subs({c30: -1.5 * 125. ** 2 / 246. / 3})
c40val = c40.subs(sol).subs({c30: -1.5 * 125. ** 2 / 246. / 3})

print(f)
# define the bounds of the minimisation, where the c20vals etc are set to fix the value
bnds = ((-400., 400.), (-400., 400.), (c20val, c20val), (c30val, c30val), (c40val, c40val))

# define initial point for minimisation
zz0 = np.array([200., 200., c20val, c30val, c40val])

# run minimisation #does not really
print("=========printing solutions from minimisation=========")
minimise(bnds, zz0)
print("======================================================")


# ###plotting tools################################

# adapted from http://matplotlib.org/examples/pylab_examples/contour_label_demo.html
class nf(float):
    def __repr__(self):
        string = '%.2e' % (self.__float__(),)
        if "e" in string:
            base, exponent = string.split("e")
            return r"$%.2f \times 10^{%i}$" % (float(base), int(exponent))
        else:
            return str


scale_pow = 2


def my_formatter_fun(x, p):
    return "%.2f" % (x * (10 ** scale_pow))


# adapted from
# http://stackoverflow.com/questions/31517156/adjust-exponent-text-after-setting-scientific-limits-on-matplotlib-axis
def format_exponent(ax, axis='y'):
    # Change the ticklabel format to scientific format
    ax.ticklabel_format(axis=axis, style='sci', scilimits=(-2, 2))

    # Get the appropriate axis
    if axis == 'y':
        ax_axis = ax.yaxis
        x_pos = 0.0
        y_pos = 1.0
        horizontalalignment = 'left'
        verticalalignment = 'bottom'
    else:
        ax_axis = ax.xaxis
        x_pos = 1.0
        y_pos = -0.05
        horizontalalignment = 'right'
        verticalalignment = 'top'

    # Run plt.tight_layout() because otherwise the offset text doesn't update
    plt.tight_layout()

    # Get the offset value
    offset = ax_axis.get_offset_text().get_text()

    if len(offset) > 0:
        # Get that exponent value and change it into latex format
        minus_sign = u'\u2212'
        expo = np.float(offset.replace(minus_sign, '-').split('e')[-1])
        offset_text = r'x$\mathregular{10^{%d}}$' % expo

        # Turn off the offset text that's calculated automatically
        ax_axis.offsetText.set_visible(False)

        # Add in a text box at the top of the y axis
        ax.text(x_pos, y_pos, offset_text, transform=ax.transAxes,
                horizontalalignment=horizontalalignment,
                verticalalignment=verticalalignment)
    return ax


def get_min_max(x, pad=0.05):
    '''
    Find min and max values such that all the data lies within 90% of
    of the axis range
    '''
    r = np.max(x) - np.min(x)
    x_min = np.min(x) - pad * r
    x_max = np.max(x) + pad * r
    return x_min, x_max


# ###Plotting 1D################
x = np.linspace(-200., 500., 201)
y = f_n(x, 0., c20val, c30val, c40val)

# plt,ax=plt.subplots()
plt.figure()
plt.plot(x, y)
# plt.xscale('symlog')


# rc('text', usetex=True)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
ax = plt.gca()
y_min, y_max = get_min_max(y)

ax.set_ylim([float(y_min), float(y_max)])

# ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(my_formatter_fun))
ax = format_exponent(ax, axis='y')
plt.savefig("./outputs/test_1D.eps")

#####Plotting 2D #####################
x1list = np.linspace(-400., 500., 201)
x2list = np.linspace(-400., 500., 201)
X, Y = np.meshgrid(x1list, x2list)
Z = f_n(X, Y, c20val, c30val, c40val)
plt.figure()
# cp = plt.contour(X, Y, Z,20,locator=ticker.LogLocator(), colors='k')
cp = plt.contour(X, Y, Z, 10)

# Recast levels to new class
cp.levels = [nf(val) for val in cp.levels]

# Label levels with specially formatted floats
# if plt.rcParams["text.usetex"]:
fmt = r'%r '
# else:
#  fmt = '%r '

# fmt = ticker.LogFormatterMathtext()
plt.clabel(cp, cp.levels, fmt=fmt, fontsize=10)
# plt.colorbar(cp)
plt.savefig("./outputs/test_2D.eps")
###############################################################################
