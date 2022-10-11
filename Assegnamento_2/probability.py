# Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
# Everyone is permitted to copy and distribute verbatim copies
# of this license document, but changing it is not allowed.

"""Definition of a class capable of derivate a pdf using some samples
of the pdf itself and capable of throwing pseudo-random number according
to the given pdf. Also possible to calculate the probability content of
some intervals in the pdf support
"""

from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

class ProbabilityDensityFunction(InterpolatedUnivariateSpline):
    """Class generating a pdf from a set of values
    """
    def __init__(self, x, y, order):
        '''Initialization of a class using some samples (x,y)
        of some pdf to interpolate the pdf itself. The class
        hold the two array of data, calculate the extremes of
        the pdf support

        Parameters
        ----------
        x : numpy array
            Values in the pdf support

        y : numpy array
            Values of the pdf calcualted in x

        order : int
            Degree of the interpolating polynomia
        '''
        self.x_array = x
        self.y_array = y
        self.x_min = np.min(x)
        self.x_max = np.max(x)
        self.order = order
        super().__init__(x, y)
        self.pdf = InterpolatedUnivariateSpline(self.x_array, self.y_array, k=self.order)

    def normalization(self):
        """Normalization of the pdf rescaling with the
        value of its integral over the entire support

        Return
        ----------
        ProbabilityDensityFunction
            Normalized pdf
        """
        integral = self.pdf.integral(self.x_min, self.x_max)
        return ProbabilityDensityFunction(self.x_array, self.y_array / integral, self.order)

    def probcontent(self, start, finish):
        """Calculate the probability of a number between
        two given values (in an interval)

        Parameters
        ----------
        start : float
            Lower bound of the selected interval

        finish : float
            Upper bound of the selected intervall

        Return
        ----------
        zone_integral : float
            The area under the pdf in the desired zone
        """
        zone_integral = self.pdf.integral(start, finish)
        return zone_integral

    def numbergen(self, num_gen):
        """Generates a selected number of random values
        distributed as the given pdf using the pdf (the
        inverse of the cpf)

        Parameters
        ----------
        num_gen : int
            Total number of generated numbers

        Return
        ----------
        An array of floats (generated numbers)
        """
        cumulant_array = np.zeros(len(self.x_array))
        for i in range(len(self.x_array)):
            cumulant_array[i] = self.pdf.integral(self.x_min, self.x_array[i])
        ppf = InterpolatedUnivariateSpline(cumulant_array, self.x_array, k=self.order)
        cumulant_value = np.random.uniform(min(cumulant_array), max(cumulant_array), num_gen)
        return ppf(cumulant_value)

if __name__ == '__main__':
    x_sample = np.linspace(0.,np.pi, 20)
    y_sample = np.sin(x_sample)
    DEG = 3.
    NUM = 10
    pdf = ProbabilityDensityFunction(x_sample, y_sample, DEG)
    pdf = pdf.normalization()
    prob_content = pdf.probcontent(0, np.pi / 2)
    genereted_values = pdf.numbergen(NUM)
    print(prob_content)
    print(genereted_values)
    plt.plot(x_sample, y_sample, 'o')
    plt.plot(x_sample, pdf(x_sample))
    plt.show()
