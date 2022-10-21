
"""Class to handle a sequence of voltage measurements at different times.
"""

import numpy
from matplotlib import pyplot as plt
from scipy import interpolate

class VoltageData:
    """Class capable to holding potential mensurament in time and
    to print and plot the data proprerly.
    """
    def __init__(self, times, voltages, voltage_errors=None):
        """Constructor of the class istance from two
        iterbale object to be converted in numpy's arrays.

        Parameters
        ----------
        times : float or arraylike
            Times recorded.

        voltages : float or arraylike
            Voltages recorded.
        """
        times_array = numpy.array(times, dtype=numpy.float64)
        voltages_array = numpy.array(voltages, dtype=numpy.float64)
        self._data = numpy.column_stack((times_array, voltages_array))
        self._spline = interpolate.InterpolatedUnivariateSpline(self.times, self.voltages, k=3)
        if voltage_errors is not None:
            voltage_errors_array = numpy.array(voltage_errors, dtype=numpy.float64)
            self._data = numpy.column_stack((self._data, voltage_errors_array))

    @classmethod
    def from_file(cls, file_path):
        """Create a class istance usign data stored in a file.

        Parameters
        ----------
        cls : function
            Constructor of the class.
        file_path : string
            Pathname of the dedired data file.

        Return
        ----------
        cls(*columns)
            An istance of the class having a variable number of
            data arrays (with errors or not).
        """
        columns = numpy.loadtxt(file_path)
        return cls(*columns)

    @property
    def times(self):
        """Defining self.times array.

        Return
        ----------
        self._data[:, 0]
            Array of times.
        """
        return self._data[:, 0]

    @property
    def voltages(self):
        """Defining self.voltages array.

        Return
        ----------
        self._data[:, 1]
            Array of voltages.
        """
        return self._data[:, 1]

    @property
    def voltage_errors(self):
        """Defining self.voltage_errors array, but only in the
        case in whitch there is a column in the data set for
        the errors on the voltages.

        Return
        ----------
        self._data[:, 2]
            Array of voltages.
        """
        try:
            return self._data[:, 2]
        except IndexError as inerr:
            error_message = 'There is no column for \'voltage_errs\'.'
            raise AttributeError(error_message) from inerr

    def rowsnumber(self):
        """Returning the leght of the data set, i.e.
        the numbers of rows.

        Return
        ----------
        self._data.shape[0]
            Int number of rows.
        """
        return self._data.shape[0]

    def columnsnumber(self):
        """Returning the number of features of
        every sample.

        Return
        ----------
        self._data.shape[1]
            Int number of columns.
        """
        return self._data.shape[1]

    def __len__(self):
        """Give length of the data set (numbers of rows).

        Return
        ----------
        self.rowsnumbers()
            Int number of samples.
        """
        return self.rowsnumber()

    def __getitem__(self, index):
        """Calling some contents of the data set with an index

        Retunr
        ----------
        self._data[index]
            Returning an array like column/row or also a
            particular float (times, voltages, errors) in
            the table
        """
        return self._data[index]

    def __iter__(self):
        """Enable iteration.
        """
        for i in enumerate(self):
            yield self._data[i, :]

    def __repr__(self):
        """Printing the rows content.
        """
        row_format = ' '.join('{}' for _ in range(self.columnsnumber()))
        return '\n'.join(row_format.format(*row) for row in self)

    def __str__(self):
        """Printing the data content in a funny way
        but only if called in the print.

        Return:

        """
        row_format = 'Row {} -> {:.1f} [s], {:.2f} [mV]'
        if self.columnsnumber() == 3:
            row_format = ' +- {:2f} [mV]'
            row_string = (row_format.format(index, *row) for index, row in enumerate(self))
        return '\n'.join(row_string)

    def __call__(self, time_input):
        """Calculate the interpolated value of voltage
        for a given time.
        """
        return self._spline(time_input)

    def plot(self, axes=None, fmt='bo', **plot_options):
        """Plotting the data (with errorbars if errors ar present).
        """
        if axes is not None:
            plt.sca(axes)
        else:
            axes = plt.figure('Voltages vs. times plot')
        try:
            plt.errorbar(self.times, self.voltages, self.voltage_errors, fmt=fmt)
        except AttributeError:
            plt.plot(self.times, self.voltages, fmt, **plot_options, label='Data plot')
        x_array = numpy.linspace(min(self.times), max(self.times), 100)
        plt.plot(x_array, self(x_array), label='Interpolation')
        plt.xlabel('Time [s]')
        plt.ylabel('Voltage [mV]')
        plt.grid(True)
        return axes
