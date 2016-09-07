"""
Class Plotting

This class provides plotting formats and functions

Public Functions:

"""

# Required Modules
import matplotlib.pyplot as plot
import numpy

# TODO: Create this class properly
# Main Class
class Plotting:

    # Public Members

    def __init__(self):
        return 0



    # Formatter to convert exp to x10 in the graphs.
    class FormatterExpTo10(float):

        def __repr__(self):
            string = '%.2e' % (self.__float__(),)
            if "e" in string:
                base, exponent = string.split("e")
                return r"$%.2f \times 10^{%i}$" % (float(base), int(exponent))
            else:
                return string

    # TODO: Jason rename this and comment as to what it does
    def my_formatter_fun(self, x, p):
        return "%.2f" % (x * (10 ** self._scale_pow))

    # Private Members
    # TODO: Comment to its purpose
    _scale_pow = 2

    # TODO: Jason comment to its purpose
    def format_exponent(ax, axis='y'):
        """
        Change the ticklabel format to scientific format

        :param axis:
        :return:
        """

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
        plot.tight_layout()

        # Get the offset value
        offset = ax_axis.get_offset_text().get_text()

        if len(offset) > 0:
            # Get that exponent value and change it into latex format
            minus_sign = u'\u2212'
            expo = numpy.float(offset.replace(minus_sign, '-').split('e')[-1])
            offset_text = r'x$\mathregular{10^{%d}}$' % expo

            # Turn off the offset text that's calculated automatically
            ax_axis.offsetText.set_visible(False)

            # Add in a text box at the top of the y axis
            ax.text(x_pos, y_pos, offset_text, transform=ax.transAxes,
                    horizontalalignment=horizontalalignment,
                    verticalalignment=verticalalignment)
        return ax


    def get_min_max_values(x, pad=0.05):
        """
        Find min and max values such that all the data lies within 90% of
        of the axis range
        :param pad:
        :return:
        """
        r = numpy.max(x) - numpy.min(x)
        x_min = numpy.min(x) - pad * r
        x_max = numpy.max(x) + pad * r
        return x_min, x_max

