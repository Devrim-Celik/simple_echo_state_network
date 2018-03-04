__author__ = "Devrim Celik"

import numpy as np
import matplotlib.pyplot as plt
import csv

#==============================================================================#

class ESN():
    """
    Echo State Network Class
    """
    def __init__(self, inSize, outSize, resSize, alpha, sparsity = 0.2):
        """
        Constructor of ESN

        Args:
            inSize (int): dimension of inputs
            outSize (int): dimension of outputs
            resSize (int): size of reservoir
            alpha (float): leaking rate
            sparsity (float): portion of connections != 0

        Attributes:
            inSize (int): dimension of inputs
            outSize (int): dimension of outputs
            resSize (int): size of reservoir
            alpha (float): leaking rate
            sparsity (float): portion of connections != 0
            Win (np.array): input to reservoir matrix [inSize+1 x resSize]
            W (np.array): reservoir to reservoir matrix [resSize x resSize]
        """
        self.inSize = inSize
        self.resSize = resSize
        self.outSize = outSize
        self.alpha = alpha
        self.sparsity = sparsity

        # +1 for Bias
        self.Win = np.random.rand(inSize+1, resSize)-0.5 # Input --> Reservoir
        self.W = np.random.rand(resSize, resSize)-0.5 # Reservoir --> Reservoir

        # First, make the reservoir connections sparse
        self.W[np.random.rand(resSize, resSize)>self.sparsity] = 0


        # calculate spectral radius and scale W, so its new spectral radius
        # is slighty smaller than 1
        spec_rad = max(abs(np.linalg.eig(self.W)[0]))
        self.W /= spec_rad/0.9



    def reservoir(self, data, new_start=True):
        """
        Args:
            data (np.array): input data points [cycles x 1]
            new_start (boolean): create new initial reservoir state?

        Attributes:
            dm (np.array): design matrix with: bias+input+
                                reservoir activity [cycles x (1+inSize+resSize)]
            R (np.array): reservoir activation [1 x resSize]
        """
        self.dm = np.zeros((data.shape[0], 1+self.inSize+self.resSize))

        # otherwise, continue from R from execution of function before
        if new_start:
            self.R = 0.1*(np.ones((1, self.resSize)) - 0.5)

        for t in range(data.shape[0]):

            u = data[t] # initialize input at timestep t

            # generate new reservoir state
            # first summand: influence of last reservoir state (same neuron)
            # second summand:
            #   first dot product: influence of input
            #   second dot product: influence of of last reservoir
            #                       state (other neurons)
            self.R = (1 - self.alpha)*self.R +                              \
                self.alpha*np.tanh( np.dot( np.hstack((1,u)), self.Win) +   \
                np.dot(self.R, self.W))

            # put bias, input & reservoir activation into one row
            self.dm[t] = np.append(np.append(1,u), self.R)

        return self.dm



    def plot_reservoir(self, path='images/', name='Plot',
                        nr_neurons=20, max_plot_cycles=100, plot_show=False):
        """
        Plotting reservoir states and their inputs from last use of .reservoir()
        are saved in a figure (and optionally displayed)

        Args:
            path (string): path of the saved plot png
            name (string): title of plot and name of saved .png
            nr_neurons (int): nr of neurons to be plotted
            max_plot_cycles (int): max number of cycles to be plotted
            plot_show (boolean): display plot?
        """
        # for plotting, separate bias, input, and real reservoir activations
        # which were saved all together in the res_history
        R = self.dm[:, -self.resSize:]
        # remove bias and get input
        R_input = self.dm[:, 1:-self.resSize]

        # check if we are below max_plot_cycles
        if R_input.shape[0] > max_plot_cycles:
            limit = max_plot_cycles
        else:
            limit = R_input.shape[0]

        plt.figure("Reservoir Activity", figsize=(20,10)).clear()
        plt.title("Reservoir Activity")
        plt.plot(R_input[:limit], color='k', label='Input Signals', linewidth=4)
        plt.plot(R[:limit, :nr_neurons], linewidth=2)
        plt.legend(loc='upper right')

        plt.savefig(path + name + '_ReservoirActivity' + '.png')
        print('\t[+]Plot saved in', path + name + '_ReservoirActivity' + '.png')

        if plot_show:
            plt.show()



    def save_dm(self, path='csv_files/', name='ESN'):
        """
        Saves current design matrix in a csv file

        Args:
            path (string): path of the saved csv file
            name (string): name of saved csv file
        """
        f = open(path + name + '.csv', 'w')
        writer = csv.writer(f)

        # create header
            # get shape of input by subracting resSize and bias
        input_shape = self.dm.shape[1]-(self.resSize+1)
        header = ['Bias']
        for i in range(input_shape):
            header.append('Input'+str(i+1))
        for i in range(self.resSize):
            header.append('Neuron'+str(i+1))
        writer.writerow(header)
        writer.writerows(self.dm)
        print('\t[+]CSV file saved in', path + name + '.csv')
