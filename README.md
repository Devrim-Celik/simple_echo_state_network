# Echo State Network
In this repository you will find a very simply echo state network, which can
be used as a scaffold for building a more sophisticated architecture. As a
dataset, the model tries to approximate the NARMA10 series, which is generated
by ```NARMA10.py``` and takes random noise as input.  
The model can be found in the ```ESN.py```file.  The ```ESN_test.py``` file
initializes an echo state network and fits it onto the NARMA10 series.  
When execute, the script will save the plot of the reservoir activity (of
neurons) and a plot of the prediction vs the real target values in the folder
```images```. Furthermore, the complete reservoir history will be saved in a
csv file in the folder ```csv_files```.

---

## Required Packages:
- [Matplotlib](https://matplotlib.org/)
- [Numpy](http://www.numpy.org/)
- [Scipy](https://www.scipy.org/)  

---

## Getting Started:
After cloning this repository and installing the required packages, simply
execute ```ESN_test.py```. You can use the following flags:

**Flags:**
- `-train_c` or `--training_cycles`: specify number of training cycles (default 4000), e.g.:
```
python3 ESN_test.py -train_c 2000
```

- `-test_c` or `--testing_cycles`: specify number of testing cycles (default 1000), e.g.:
```
python3 ESN_test.py -test_c 500
```

- `-rs` or `--resSize`: specify reservoir size (default 1000), e.g.:
```
python3 ESN_test.py -rs 200
```

- `-a` or `--alpha`: specify leaking rate (default 0.8), e.g.:
```
python3 ESN_test.py -a 0.2
```

- `-s` or `--show`: show plots e.g.:
```
python3 ESN_test.py -s
```

---

## NARMA10 Prediction
![NARMA10 Prediction](https://github.com/MistySheep/Echo_State_Network/blob/master/images/prediction.png)

---

## Reservoir Activity Sample
![Reservoir Activity Sample](https://github.com/MistySheep/Echo_State_Network/blob/master/images/reservoir_act.png)
