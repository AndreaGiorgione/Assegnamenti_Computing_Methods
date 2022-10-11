
"""
"""

from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
import random

class ProbabilityDensityFunction(InterpolatedUnivariateSpline):
    """
    """
    def __init__(self, x, y, a):
        """
        """
        self.xarray = x
        self.yarray = y
        self.xmin = np.min(x)
        self.xmax = np.max(x)
        self.order = a
        super().__init__(x, y)
        self = InterpolatedUnivariateSpline(self.xarray, self.yarray, k=self.order)
        
    def normalization(self):
        """
        """
        I = self.integral(self.xmin, self.xmax)
        return ProbabilityDensityFunction(self.xarray, self.yarray / I, self.order)
    
    def probcontent(self, start, finish):
        """
        """
        I = self.integral(start, finish)
        return I

    def numbergen(self):
        """
        """
        q = np.zeros(len(self.xarray))
        for i in range(len(self.xarray)):
            q[i] = self.integral(self.xmin, x[i])
        new = InterpolatedUnivariateSpline(q, x, k=self.order)
        q = random.uniform(min(q), max(q))
        return new(q)

if __name__ == '__main__':
    x = np.linspace(0.,np.pi, 20)
    y = np.sin(x)
    f = ProbabilityDensityFunction(x, y, 3)
    g = f.normalization()
    I = g.probcontent(0.2, 0.4)
    value = np.zeros(10000)
    for i in range(10000):
        value[i] = g.numbergen()
    plt.hist(value, bins=30, weights=np.full(len(value), 1/1000))
    plt.plot(x, y, 'o')
    plt.plot(x, g(x))
    plt.show()
