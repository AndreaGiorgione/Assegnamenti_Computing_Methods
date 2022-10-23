
"""Test for the VoltageData class in voltage_data_full.py.
"""

import unittest
import numpy
from matplotlib import pyplot
from voltage_data_full import VoltageData

class TestVoltageData(unittest.TestCase):
    """Class containing a test for every functionality
    ot the class VoltageData.
    """
    def setUp(self, sample_size=10):
        """Creation from data of the data array and the
        VoltageData object.
        """
        # file_path = 'sample_data_file_with_errors.txt'
        # self.target = VoltageData.from_file(file_path)
        # self.times, self.voltages, self.voltage_errors = numpy.loadtxt(file_path)
        self.sample_size = sample_size
        self.times = numpy.linspace(0., 2., self.sample_size)
        self.voltages = numpy.random.uniform(0.5, 1.5, self.sample_size)
        self.voltage_errors = numpy.repeat(0.05, self.sample_size)
        self.target = VoltageData(self.times, self.voltages, self.voltage_errors)

    def test_dataset(self):
        """Checking the right shape of data set
        """
        self.assertEqual(len(self.times), len(self.voltages))
        self.assertEqual(len(self.times), len(self.voltage_errors))

    def test_constructor(self):
        """Test the initialization of the VoltageData object.
        """
        self.assertEqual(self.target._data.shape, (len(self.times), 3))

    def test_columnsnumber(self):
        """Checking the number of features of the dataset in target.
        """
        self.assertEqual(self.target.columnsnumber(), 3)

    def test_rowsnumber(self):
        """Cheking the length of the dataset in target.
        """
        self.assertEqual(self.target.rowsnumber(), len(self.times))

    def test_len(self):
        """Testing the __len__ method (length of the columns)
        in VoltageData.
        """
        self.assertEqual(len(self.target), len(self.times))

    def test_getitem(self):
        """Test the use of index to get a particular number of
        the dataset in target.
        """
        self.assertEqual(self.target[1, 0], self.times[1])
        self.assertEqual(self.target[self.sample_size // 2, 1], \
                         self.voltages[self.sample_size // 2])
        self.assertEqual(self.target[self.sample_size - 1, 2], \
                         self.voltage_errors[self.sample_size - 1])

    def test_iter(self):
        """Testing the right behaviour of the method __iter__
        (iteration over an index) in VoltageData.
        """
        for i, entry in enumerate(self.target):
            self.assertEqual(entry[0], self.times[i])
            self.assertEqual(entry[1], self.voltages[i])
            self.assertEqual(entry[2], self.voltage_errors[i])

    def test_printing(self):
        """Check the format of returned string of __str__
        and __repr__ methods.
        """
        print(self.target)
        print(repr(self.target))
        self.assertEqual(type(repr(self.target)), str)

    def test_interpolation(self):
        """Test the fitting of the spline respect to data.
        """
        delta = abs(self.target(self.times) - self.voltages)
        self.assertTrue((delta < 1e-12).all())

    def test_plot(self):
        """Checking graphical results.
        """
        self.target.plot(fmt='ko', markersize=6, label='normal voltage')
        pyplot.show()

if __name__ == "__main__":
    unittest.main()
