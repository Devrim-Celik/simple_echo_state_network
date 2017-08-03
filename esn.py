#####################################################################
############################## IMPORTS ##############################
#####################################################################
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg



#####################################################################
############################ PARAMETERS #############################
#####################################################################

######################
####### CYCLES #######
######################
train_cycles = 4000 # Amount of Training Cycles
validate_cycles = 1000 # Amount of Validation Cycles
warmLen = 100 # Amount of Warmup Cycles
train_cycles += warmLen
error_cycles = 500 # Amount of Cycles for calculating the Error

######################
####### SIZES ########
######################
inSize = 1 # Amount of Input Features
outSize = 1 # Amount of Output Features
resSize = 1000 # Size of reservoir
a = 0.3 # Leaking Rate

np.random.seed(42)



#####################################################################
######################### DATA AQUISITION ###########################
#####################################################################
data = np.loadtxt('Mackey.txt')

######################
### PLOTTING DATA ####
######################
plt.figure(5)
plt.plot(data[0:100])
plt.title('Sample of Input Data')



#####################################################################
################## WEIGHT MATRICE INITIALIZATION ####################
#####################################################################
# +1 for bias / uniform distributed between -0.5 and 0.5
Win = np.random.rand(inSize+1, resSize)-0.5 # Input --> Reservoir
W = np.random.rand(resSize, resSize)-0.5 # Reservoir --> Reservoir

######################
### SPECTRAL RADIUS ##
### & NORMALIZATION ##
######################
# Set Spectral Radius
print('[*]Computing spectral radius...')
spec_rad = max(abs(scipy.linalg.eig(W)[0]))
print('[+]Done.')
# TODO why 1.25
#W *= 1.25/spec_rad
W /= spec_rad

######################
### DESIGN MATRIX ####
######################
# dimension (cycles x bias + input_features + reservoir_size)
X = np.zeros((train_cycles - warmLen, 1+inSize+resSize))

######################
### TARGET MATRIX ####
######################
# TODO why plus 1
# idea: so we dont take first value, because....
# idea2: because this is for generation, we try to predict the next values!!!
Yt = data[warmLen+1:train_cycles+1]



#####################################################################
############################ TRAINING ###############################
#####################################################################
print('[*]Training...')

######################
### RESERVOIR RUN ####
######################
# TODO why plus 1
# idea: so we dont take first value, because....
# idea2: because this is for generation, we try to predict the next values!!!
x = np.zeros((1,resSize)) # Memory to save Reservoir State
for t in range(train_cycles):
    u = data[t]
    # generate new State
    x = (1-a)*x + a*np.tanh( np.dot( np.hstack((1,u)), Win) + np.dot(x, W))

    # if we are not in the warmup phase, save State
    if t >= warmLen:
        # TODO ugly
        temp = np.append(1,u)
        X[t-warmLen] = np.append(temp, x)

##########################
### LINEAR REGRESSION ####
##########################
Wout = np.dot( np.linalg.pinv(X), Yt ) # via Pseudoinverse

print('[+]Done.')



#####################################################################
############################ TESTING ################################
#####################################################################
Predictions = np.zeros((validate_cycles, outSize))

u = data[train_cycles] # continue where you left of for generating next output (+1 implicit)

print('[*]Testing...')

for t in range(validate_cycles):

    x = (1-a)*x + a*np.tanh( np.dot( np.hstack((1,u)), Win) + np.dot(x, W))


    # TODO
    temp = np.append(1,u)
    y = np.dot(np.append(temp,x), Wout) # generate Prediction using Linear Regression Weights

    Predictions[t,:] = y # add to Predictions

    # For a Predictive Model
    #u = data[train_cycles+t+1]
    # For a Generative model
    u = y

print('[+]Done.')

##########################
######### ERROR ##########
##########################
MSE = sum( np.square( data[train_cycles+1:train_cycles+error_cycles+1] - Predictions[:error_cycles] ) ) / error_cycles
print('MSE = ' + str( MSE ))
