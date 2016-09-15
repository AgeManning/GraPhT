"""
Class Potential

This class performs calculations on arbitrary potentials.

Public Functions:
    minimize    : Minimize a generic potential
TODO: Complete this

"""

# Required Modules
import numpy
import sympy as sy
from sympy.matrices import hessian
from sympy.solvers import solve
# Currently Unused
from scipy.optimize import minimize
from scipy import optimize
from sympy.utilities.lambdify import lambdify


# Main Class
class Potential:
    # Member Variables

    # Constructor
    def __init__(self, _potential_parameters, _constants_values=None):
        # TODO: Validate _potential_parameters

        self.potential  = _potential_parameters["potential_form"]
        self._fields    = _potential_parameters["fields"]
        self._constants = _potential_parameters["constants"]
        self._masses    = _potential_parameters["masses"]
        self._vevs      = _potential_parameters["vevs"]
        self._unknown_constants = _potential_parameters["unknown_constants"]
        self._known_constants = _potential_parameters["known_constants"]
        self._known_constants_vals = _potential_parameters["known_constants_vals"]

        # Create a functional form of the potential
        all_parameters = self._fields + self._constants
        self.functional_potential = lambdify(all_parameters, self.potential, modules='numpy')

        # Define a list of constants as a function of others
        self._constant_reduction = self._impose_minima_constraints()

        # Constrain the potential
        self._constrained_potential = self.potential.subs(self._constant_reduction)


        # If we have constants values, set them
        if _constants_values is not None:
            self.set_constants(_constants_values)

    # Other public functions

    #the one immediately below is a helper function
    def func_to_minimise(self, x):
        """
        helper function
        :return: functonal potential
        """
        freduced_potential = lambdify(self._fields, self._constrained_potential, modules='numpy')
        return freduced_potential(*tuple(x))

    # TODO: possibly remove initial as argument in the function below
    def minimise(self, bounds, initial):
        """
        Minimizes a generic potential. Returns the global minimum.
        :param Potential        : A python function representing the potential
        :param Bounds           : Limits/Range over which to minimize
        :param Temperature      : Temperature
        :param InitialPoint    : Initial guess for minimizes
        :return: The global minimum point. Minimize class
        """

        # TODO: Implement this properly, currently does tree level potential
        print("minimising potential:", self._constrained_potential)

        # First use brute force minimisation
        min_config=optimize.brute(self.func_to_minimise,bounds)
        #print("minima field config:", min_config)

        # Use the brute force results to get more accurate results
        # Nelder-Mead used because no gradient info required, tolerence specified
        min_config_acc = minimize(self.func_to_minimise,min_config, method='Nelder-Mead',tol=1e-12)
        # TODO: check if other minimisation methods are better, in general graidient based methods are better
        #min_config = minimize(freduced_potential, initial, method='SLSQP', jac=jac_v, bounds=bounds, tol=1e-12)
        #print(min_config_acc)

        # TODO: probably add field configuration and minimum value as attributes/instance of the class
        minima = {}
        minima['config'] = min_config_acc.x
        minima['val']    = min_config_acc.fun

        return minima

    def remaining_vars(self):
        """
        Returns the remaining variables after imposing the minima constraints
        :return: tuple containing remaining variables
        """
        impose_constraints = self._impose_minima_constraints()
        reduced_potential = self.potential.subs(impose_constraints)
        remaining_variables = tuple(reduced_potential.free_symbols - set(self._fields))
        return remaining_variables

    def set_constants(self, constants):
        ## TODO: We may want to set a class variable to store this function as opposed to returning it
        """
        Sets numeric values for the constants in the constrained potential.
        :param constants: Dictionary containing constants
        :return: The value of the potential with unkowns specified
        """
        return self._constrained_potential.subs(constants)

    def numeric_const_values(self,constant_values):
        """
        # TODO: Make sure this doesn't fail. I.e Check we have entered enough numbers
        Gets the numeric values of constants when fixing the unkown variables
        :param constants:   Dictionary fixing remaining vars
        :return:            Numeric values of the constants
        """
        numeric_dict = {}
        for constant in self._constant_reduction:
            # Loops through the constant dependence list, i.e {c1: 3c2 + 4 c4, c3: c2} and subs known values
            # for c2, c4 etc
            numeric_dict[constant] = self._constant_reduction[constant].subs(constant_values)
        return numeric_dict



    # Private Variables

    # Private Functions

    def _calculate_masses(self):
        """
        Calculates the hessian of the fields and diagonalizes it to obtain the masses
        :return: <Insert Return type and object here>
        """
        h = hessian(self.potential, self._fields)
        (p, d) = h.diagonalize()
        return numpy.diagonal(d)

    def _impose_minima_constraints(self):
        """
        Imposes the minima constraints on the potential. Optional for 1-loop
        :param unknown_constants:
        :return:
        """
        vevtuple = solve(numpy.subtract(self._fields, self._vevs))

        # First Derivative
        first_derivative = ([sy.diff(self.potential, i).subs(vevtuple) for i in self._fields])
        # TODO: Check this logic - where a field has only mass terms
        # Diagaonalized matrices
        diagonal_hessian = ([m.subs(vevtuple) for m in self._calculate_masses()])  # subs only work on syMatrices

        lhs = tuple(first_derivative) + tuple(diagonal_hessian) + tuple(self._known_constants)
        rhs = tuple([0. for i in self._fields]) + tuple([m ** 2 for m in self._masses])  + tuple(self._known_constants_vals)

        return solve(list(numpy.subtract(lhs, rhs)), self._constants, check=True)  # ,set=True
        #return solve(list(numpy.subtract(lhs, rhs)), self._unknown_constants, check=True)  # ,set=True

    def _quantum_corrections(self,SM=True):
        """
        Use on-shell renormalisation with cutoff regularisation so that the
        CW contribution does not shift the derivatives of the potential
        see e.g. 1409.0005
        :return:
        """
        # TODO: are we going to let the user decide the field-dependent masses?
        # TODO: Need to think about alternate SM structure but no new particles

        CW= 0
        CW+= 1./64.

        if SM==True:
            CW+= 1.

    def _thermal_correction(self):
       """
       Add thermal correction
       :return:
       """



    """ Old Private Functions. May use in the future...
    # Optimization
    # Build Jacobian if require to be used in function "minimise:
    jac_f = [f.diff(x) for x in xx]
    jac_fn = [lambdify(xx, jf, modules='numpy') for jf in jac_f]
    # Jacobian Helper for receiving vector parameters
    def _jac_v(zz, jac_fn):
    return numpy.array([jfn(zz[0], zz[1], zz[2], zz[3], zz[4]) for jfn in jac_fn])

    """
