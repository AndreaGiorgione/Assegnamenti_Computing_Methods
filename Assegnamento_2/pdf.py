# -*- coding: utf-8 -*-
#
# Copyright (C) 2022 Andrea Giorgione (andreagiorgione98@gmail.com)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Definition of a class capable of derivate a pdf using some samples
of the pdf itself and capable of throwing pseudo-random number according
to the given pdf. The class permitts also to calculate the probability
content of some intervals in the pdf support.
"""

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

class ProbabilityDensityFunction(InterpolatedUnivariateSpline):
    """Class generating a pdf from a set of values
    """
    def __init__(self, x, y, order):
        '''Initialization of a class using some samples (x,y)
        of some probability density function (pdf) to interpolate
        the pdf itself. The resulting gunction is normalized and
        then used for the contruction of the cumulative density
        function (cdf) and the probability point function (ppf).

        Parameters
        ----------
        x : array-like
            Values in the pdf support.

        y : array-like
            Values of the pdf calcualted in x.

        order : int
            Degree of the interpolating polynomial.
        '''
        # Normalization redefining the pdf values (change ony if needed).
        norm = InterpolatedUnivariateSpline(x, y, k=order).integral(x[0], x[-1])
        y /= norm
        super().__init__(x, y, k=order) # Initailizing the pdf.
        # Building the cdf interpolating the integrals of the inputs.
        ycdf = np.array([self.integral(x[0], xcdf) for xcdf in x])
        self.cdf = InterpolatedUnivariateSpline(x, ycdf, k=order)
        # Building the ppf inverting (by 90 deggre) the cdf.
        # Making sure there are no equal values of x (diverging coefficient).
        xppf, ippf = np.unique(ycdf, return_index=True)
        yppf = x[ippf] # Acoording arrays.
        self.ppf = InterpolatedUnivariateSpline(xppf, yppf, k=order)

    def prob(self, inf_x, sup_x):
        """Calculate the probability of a number between
        two given values using cfd (in an interval).

        Parameters
        ----------
        inf_x : float
            Lower bound of the selected interval.

        sup_x : float
            Upper bound of the selected intervall.

        Return
        ----------
        Float
            The area under the pdf in the desired interval.
        """
        return self.cdf(sup_x) - self.cdf(inf_x)

    def rnd(self, num_gen):
        """Generates a selected number of random values
        distributed according to the given pdf using
        the ppf (the inverse of the cpf).

        Parameters
        ----------
        num_gen : int
            Total number of generated values.

        Return
        ----------
        Float or array-like
            Generated values.
        """
        return self.ppf(np.random.uniform(size=num_gen))
