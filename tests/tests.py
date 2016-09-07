"""
This function defines modules on which to test our code.

Run arbitrary code within here.

"""
import sys
# Allow us to import our classes
sys.path.append('../classes')
# Import custom classes
from potential import Potential
# Import standard modules - Ideally we shouldn't need these here
import sympy


def run_test1():
    # TODO: Code goes here
    print("Here there be dragons....")

def potential_test():
    # We can use this to test the potential
    print("Testing the potential with stuff...")


def run_test(configData):
    # Example with config Data
    if not configData:
        print("Using configuration file: ",configData)


    # Set up our potential class
    # TODO: Convert potential from config file

    # We will pass a dictionary that we will get from the config file. For now we construct it manually

    potential_parameters = {}

    # define fields and constants for the time being
    # define constants
    const_2, const_3, const_4 = sympy.symbols('c2,c3,c4')
    constants = (const_2, const_3, const_4)
    # define fields
    field_1, field_2 = sympy.symbols('f1,f2')
    fields = (field_1, field_2)
    # Define the potential
    potential_form = const_2 * field_1 ** 2 + const_3 * field_1 ** 3 + const_4 * field_1 ** 4  # + x2**4
    # Vevs and masses
    vevs = (246., 0.)
    masses = (0., 125.)  # some ordering may be required to match diagonalisation

    # TODO: Are minimization bounds going to be derivived or inputs?. If inputs, put here and feed into potential parameters
    # Bounds and Initial Points

    # Build the dictionary
    potential_parameters['fields'] = fields
    potential_parameters['constants'] = constants
    potential_parameters['potential_form'] = potential_form
    potential_parameters['vevs'] = vevs
    potential_parameters['masses'] = masses
    potential_parameters['unknown_constants'] = (const_2, const_4)

    # Create our potential class
    # This stores all info about the potential - See the class
    potential = Potential(potential_parameters)
    # potential currently cannot handle pure mass terms

    # solve the constants from 1st derivative and hessian of potential
    # This will be private function - But for testing lets run it
    print("=======printing solutions to solving constants========")
    print(potential._constant_dependence)
    print("======================================================")

    # ####working out the free-parameters######
    print("Remaining Variables :", potential.remaining_vars())

    # Only printing for testing. Private member
    print("Constrained Potential :", potential._constrained_potential)
    # Set some constants
    constant_val = {'c3': -1.5 * 125. ** 2 / 246. / 3}
    print("Potential with c3 Fixed :", potential.set_constants(constant_val))

    # Get Numeric Values of Constants given fixed unknown variables
    print("Numeric Values for Constants with c3 fixed :", potential.numeric_const_values(constant_val))


    # define the bounds of the minimisation, where the c20vals etc are set to fix the value
    # define initial point for minimisation
    # Initial point for finding minima

    # Bounds for the minimization
    # TODO: Implement minimization with bounds and initial point.
    # TODO: Implementation is done in the class not here!
    # TODO: Bounds can be class members and set up from config. Or dervived in the constructor
    # bnds = ((-400., 400.), (-400., 400.), (c20val, c20val), (c30val, c30val), (c40val, c40val))
    # bounds = < insert some points >

    #initial_minimization_point = numpy.array([200., 200., c20val, c30val, c40val])
    # run minimisation #does not really
    (bounds, initial_point) = (0,0)
    print("=========printing solutions from minimisation=========")
    print(potential.minimise(bounds, initial_point))
    print("======================================================")

    # Plotting
    """ Leave for Now... will fix soon
    ## Just for compatibility
    # ###Plotting 1D################
    # ALL PLOTTING BELOW
    # Sets up points
    x = np.linspace(-200., 500., 201)
    y = f_n(x, 0., c20val, c30val, c40val)

    plt.figure()
    plt.plot(x, y)

    # Rescale y-axis
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax = plt.gca()
    y_min, y_max = get_min_max(y)
    ax.set_ylim([float(y_min), float(y_max)])

    # Make the top scale and exponent
    ax = format_exponent(ax, axis='y')
    plt.savefig("./outputs/test_1D.eps")

    #####Plotting 2D #####################
    # Splitting into numbers into array
    x1list = np.linspace(-400., 500., 201)
    x2list = np.linspace(-400., 500., 201)
    X, Y = np.meshgrid(x1list, x2list)
    Z = f_n(X, Y, c20val, c30val, c40val)

    plt.figure()


    # Set up contour plot
    cp = plt.contour(X, Y, Z, 10)

    # Recast levels to new class
    cp.levels = [nf(val) for val in cp.levels]

    # Label levels with specially formatted floats
    fmt = r'%r '

    plt.clabel(cp, cp.levels, fmt=fmt, fontsize=10)
    plt.savefig("./outputs/test_2D.eps")
    ###############################################################################
"""
