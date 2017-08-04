"""
File name: ESN.py
    Author: Devrim Celik
    Date created: 08/04/2017
    Date last modified: 08/04/2017
    Python Version: 3.6
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import NARMA10
import error_metrics as em

# TODO add generative
class ESN():
    """
    Echo State Network Class
    """
    def __init__(self, inSize, outSize, resSize, alpha):
        """
        Constructor of ESN

        Args:
            inSize (int): dimension of inputs
            outSize (int): dimension of outputs
            resSize (int): size of reservoir
            alpha (float): leaking rate

        Attributes:
            inSize (int): dimension of inputs
            outSize (int): dimension of outputs
            resSize (int): size of reservoir
            alpha (float): leaking rate
            Win (np.array): input to reservoir matrix [inSize+1 x resSize]
            W (np.array): reservoir to reservoir matrix [resSize x resSize]
        """
        self.inSize = inSize
        self.resSize = resSize
        self.outSize = outSize
        self.alpha = alpha

        self.Win = np.random.rand(inSize+1, resSize)-0.5 # Input --> Reservoir
        self.W = np.random.rand(resSize, resSize)-0.5 # Reservoir --> Reservoir

        print('[*]Computing spectral radius...')
        spec_rad = max(abs(scipy.linalg.eig(self.W)[0]))
        print('\t[+]Done.')
        # TODO why 1.25
        self.W *= 1.25/spec_rad
        #self.W /= spec_rad



    def train(self, data, target, train_cycles, warmup_cycles):
        """
        Initiate training phase of ESN

        Args:
            data (np.array): input data points [train_cycles+warmup_phase x 1]
            target (np.array): target values [train_cycles+warmup_phase x 1]
            train_cycles (int): amount of training cycles/inputs
            warmup_cycles (int): amount of warmup cycles/inputs

        Attributes:
            DM_train (np.array): design matrix [train_cycles x (1+inSize+resSize)]
            u_train (np.array): input data points [train_cycles+warmup_phase x 1]
            Y_train (np.array): target values [train_cycles+warmup_phase x 1]
            R (np.array): czrrent reservoir state [1 x resSize]
            Wout (np.array): input&reservoir to output matrix [(1+inSize+resSize) x outSize]
        """
        self.DM_train = np.zeros((train_cycles, 1+self.inSize+self.resSize))
        self.u_train = data
        self.Y_train = target[warmup_cycles:] # exclude warmup target values
        # TODO try different initializing values
        self.R = np.zeros((1, self.resSize)) # reservoir State

        print('[*]Training...')

        for t in range(warmup_cycles + train_cycles):

            u = self.u_train[t] # initialize input at timestep t

            # generate new reservoir state
            # first summand: influence of last reservoir state (same neuron)
            # second summand:
            #   first dot product: influence of input
            #   second dot product: influence of of last reservoir state (other neurons)
            self.R = (1 - self.alpha)*self.R + self.alpha*np.tanh( np.dot( np.hstack((1,u)), self.Win) + np.dot(self.R, self.W))

            # in case we are not in warmup, save values in design matrix
            if t >= warmup_cycles:
                # TODO ugly
                temp = np.append(1,u)
                self.DM_train[t-warmup_cycles] = np.append(temp, self.R)

        # calculate Wout/("weights") with linear regression via pseudoinverse
        self.Wout = np.dot( np.linalg.pinv(self.DM_train), self.Y_train )

        print('\t[+]Done.')



    def inference(self, data, target, test_cycles):
        """
        Use trained ESN with given input (as a predictive model)

        Args:
            data (np.array): input data points [test_cycles x 1]
            target (np.array): target values (for later plotting)
            test_cycles (int): amount of testing cycles/inputs

        Attributes:
            u_test (np.array): input data points [test_cycles x 1]
            Y (np.array): target values (for later plotting)
            Yhat (np.array): predictions of ESN [test_cycles x outSize]
            R_history (np.array): history of reservoir states [test_cycles x resSize]

        Returns:
            Yhat
        """

        self.u_test = data
        self.Y = target
        self.Yhat = np.zeros((test_cycles, self.outSize))
        self.R_history = np.zeros((test_cycles, self.resSize))

        print('[*]Testing...')

        for t in range(test_cycles):
            u = self.u_test[t]

            self.R  = (1 - self.alpha)*self.R + self.alpha*np.tanh( np.dot( np.hstack((1,u)), self.Win) + np.dot(self.R, self.W))

            # save R in history
            self.R_history[t] = self.R

            # caclulate output
            # TODO ugly
            temp = np.append(1,u)
            self.Yhat[t] = np.dot( np.append(temp, self.R), self.Wout)

        print('\t[+]Done.')

        return self.Yhat

    def plot(self, cycles, name='Plot', plot_input=True, plot_reservoir=True, nr_neurons=5, plot_target=True, plot_predictions=True, path='pictures/'):
        """
        Plotting after inference with trained ESN

        Args:
            path (string): path of the saved plot png
            title (string): title of plot (in picture)
            cycles (int): nr of cycles to be plotted
            plot_input (boolean): plot input values?
            plot_reservoir (boolean): plot reservoir values?
                nr_neurons (int): number of reservoir neurons to be plotted
            plot_target (boolean): plot target values?
            plot_predictions (boolean): plot predicted values?
        """

        if  plot_target or plot_predictions:

            plt.figure(figsize=(20,10))
            plt.title(name)

            if plot_target:
                plt.plot(self.Y[:cycles], color='b', label='Target Values', linewidth=6)
            if plot_predictions:
                plt.plot(self.Yhat[:cycles], color='r', label='Predictions of ESN', linewidth=2, linestyle='--')
            if plot_input:
                plt.plot(self.u_test[:cycles], color='k', label='Input Signals', linestyle=':')
            if plot_reservoir:
                plt.plot(self.R_history[:cycles, :nr_neurons], linewidth=0.33, label='Neuron X', )

            plt.legend(loc='upper right')
            plt.savefig(path + name + '.png')
            print('Plots saved in', path + name + '.png')
            plt.show()

def default_test_NARMA10(plot_path='pictures/', plot_name='default_test_NARMA10', plot_cycles=100, train_cycles=4000, test_cycles=1000, warmup_cycles=100, alpha=0.8, resSize=1000):

    # Data
    data, Y = NARMA10.getData(warmup_cycles+train_cycles+test_cycles)
    data_train, Y_train = data[:train_cycles+warmup_cycles], Y[:train_cycles+warmup_cycles]
    data_test, Y_test = data[warmup_cycles+train_cycles:], Y[warmup_cycles+train_cycles:]

    # Network
    Echo = ESN(1, 1, resSize, alpha)
    Echo.train(data_train, Y_train, train_cycles, warmup_cycles)
    Yhat = Echo.inference(data_test, Y_test, test_cycles)

    # Error
    NRMSE = em.NRMSE(Y_test, Yhat)

    # Plotting & Saving
    plot_name += '_NRMSE=' + str(NRMSE)
    Echo.plot(plot_cycles, path=plot_path, name=plot_name)

default_test_NARMA10()
