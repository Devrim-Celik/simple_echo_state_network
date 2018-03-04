__author__ = "Devrim Celik"

import matplotlib.pyplot as plt
import numpy as np
import argparse

import NARMA10
import ESN

#==============================================================================#
# testing function
def default_test_NARMA10(plot_path='images/', plot_name='test_NARMA10',
                        inSize=1, outSize=1, train_cycles=4000, test_cycles=1000,
                        alpha=0.8, resSize=1000, plot_show=False):

    # Get Data
    data, Y = NARMA10.getData(train_cycles+test_cycles)
    data_train, Y_train = data[:train_cycles], Y[:train_cycles]
    data, Y = NARMA10.getData(train_cycles+test_cycles)
    data_test, Y_test = data[train_cycles:], Y[train_cycles:]

    # Reservoir & Training
        # setup reservoir
    Echo = ESN.ESN(inSize, outSize, resSize, alpha)
        # get reservoir activations for training data
    RA_Train  = Echo.reservoir(data_train)
        # caclulate output matrix via moore pensore pseudoinverse (linear reg)
    Wout = np.dot(np.linalg.pinv(RA_Train), Y_train )
        # get reservoir activation for test data
    RA_Test = Echo.reservoir(data_test, new_start=True)
        # calculate predictions using output matrix
    Yhat = np.dot(RA_Test, Wout)

    # Calculate Error
        # we throw away the first 50 values, cause the system needs
        # enough input to being able to predict the NARMA10, since it is a
        # delayed differential equation
    NRMSE = np.sqrt(np.divide(                          \
        np.mean(np.square(Y_test[50:]-Yhat[50:])),   \
        np.var(Y_test[50:])))

    #print(NRMSE)

    # Plotting & Saving
    plot_name += '_NRMSE={0:.4f}'.format(NRMSE)
    Echo.plot_reservoir(path=plot_path, name=plot_name, plot_show=plot_show)

    Echo.save_dm(name=plot_name)

    # Prediction Plot
    plt.figure('Prediction', figsize=(14,7)).clear()
    plt.yscale('log')
    plt.plot(Y_test, color='red', linewidth=5, label='Target Value')
    plt.plot(Yhat, color='blue', linestyle="--", linewidth=1, label='ESN Prediction')
    plt.legend()
    plt.savefig(plot_path + plot_name + '.png')
    print('\t[+]Plot saved in', plot_path + plot_name + '.png')
    if plot_show:
        plt.show()

    return NRMSE

#==============================================================================#

if __name__=='__main__':
    # default values
    dv = {'train_c': 4000, 'test_c':1000, 'resSize':1000, 'alpha':0.8}

    # define parser & its arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-train_c', '--training_cycles', nargs=1, help="Number of Training Cycles", type=int)
    parser.add_argument('-test_c', '--testing_cycles', nargs=1, help="Number of Testing Cycles", type=int)
    parser.add_argument('-rs', '--size', nargs=1, help="Size of Reservoir", type=int)
    parser.add_argument('-a', '--alpha', nargs=1, help="Leaking Rate", type=float)
    parser.add_argument('-s', '--show', const=True, nargs="?", help="Show Plot")
    # parse
    args = parser.parse_args()
    # collect values
    arguments = [args.training_cycles, args.testing_cycles, args.size, args.alpha]

    # check if default should be replaced
    for nr, key in enumerate(dv.keys()):
        if arguments[nr] != None:
            dv[key] = arguments[nr][0]
    # add show separatly
    dv['show'] = args.show!=None

    # test
    default_test_NARMA10(train_cycles=dv['train_c'], test_cycles=dv['test_c'], resSize=dv['resSize'], alpha=dv['alpha'], plot_show=dv['show'])
