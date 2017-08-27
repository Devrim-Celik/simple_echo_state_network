# Echo State Network

## Requirements:
- [Matplotlib](https://matplotlib.org/)
- [Numpy](http://www.numpy.org/)
- [Scipy](https://www.scipy.org/)  

## Instructions:

### Basic NARMA10 Test
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
