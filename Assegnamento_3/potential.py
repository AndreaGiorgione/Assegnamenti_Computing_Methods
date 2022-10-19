import numpy
from scipy import interpolate
from matplotlib import pyplot as plt

class VoltageData:
    """
    """
    def __init__(self, times, voltages):
        """
        """
        times = numpy.array(times, dtype=numpy.float64)
        voltages = numpy.array(voltages, dtype=numpy.float64)
        # if len(self.times) is not len(self.voltages):
        # raise ValueError('No')
        self.data = numpy.column_stack([times, voltages])
        self._spline = interpolate.InterpolatedUnivariateSpline(self.times, self.voltages, k=3)

    @classmethod
    def from_file(cls, data_path):
        times, voltages = numpy.loadtxt(data_path, unpack=True)
        return cls(times, voltages)

    @property
    def times(self):
        return self.data[:, 0]

    @property
    def voltages(self):
        return self.data[:, 1]

    def __getitem__(self, index): # Variable index could be anything
        return self.data[index] # All cases works thanks to numpy

    def __len__(self):
        return len(self.data)

    def __iter__(self): # Sholud iteret for rows
        return iter(self.data)

    def __str__(self):
        # output_str = ''
        # for i, row in enumerate(self):
        # line = f'{i} -> {row[0]:1f}, {row[1]:2f}\n'
        # output_str += line
        #return output_str
        header = 'row ->  time (s), voltage (mV)\n'
        return header + '\n'.join([f'{i} -> {row[0]:1f}, {row[1]:2f}' \
                            for i, row in enumerate(self)])
    
    def __repr__(self):
        # return str(self.data)
        return '\n'.join([f'{row[1]}'for row in self])

    def __call__(self, time):
        return self._spline(time)

    def plot(self, ax=None, **plot_opts):
        if ax is None:
            plt.figure('Voltage plot')
        else:
            plt.sca(ax)
        plt.plot(self.times, self.voltages, **plot_opts, label='Data plot')
        x = numpy.linspace(min(self.times), max(self.times), 100)
        plt.plot(x, self(x), label='Interpolation')
        plt.xlabel('Times (s)')
        plt.ylabel('Voltages (mV)')
        plt.grid(True)
        plt.legend()

if __name__ == '__main__':
    # t = [1., 2., 3., 4. , 5., 6.]
    # v = [10., 20., 30., 50., 90., 130.]
    # t, v = numpy.loadtxt('sample_data_file.txt', comments='#', unpack=True)
    # vdata = VoltageData(t, v)
    vdata = VoltageData.from_file('sample_data_file.txt')
    print(vdata.times, vdata.voltages)
    print(vdata[0, 0])
    # assert vdata[0, 0] == 1.
    print(len(vdata))
    for element in vdata:
        print(element)
    print(vdata)
    print(repr(vdata))
    print(vdata(1.5))
    vdata.plot(marker='o', linestyle='--', color='r')
    plt.show()
