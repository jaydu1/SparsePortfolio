# High-Dimensional Portfolio Selecton with Cardinality Constraints


# Folder and Files

- `spo/`: package implementing the algorithm.
- `requirements.txt`: package requirements for experiments.
- `experiment/`: scripts to run all experiments in the paper.
- `data/`: preprocessed data obtained from Wharton CRSP historical stock return database. After putting the raw data input `data/raw/`, run the script `data/preprocess.py` to preprocess the raw data.
- `plotting.py`: a script to reproduce all the figures.

# Requirements

We use Python 3 for our experiment.
Please refer to `requirements.txt`, and use `pip` or `conda` to create a virtual environment with required packages installed.

For obtain final results, do the following steps in the current workspace:

- Create empty folders `results/` and `img/`. 
- Run all scipts in the folder `experiment/`, and the results will be stored in `.npy` files.
- Use `plotting.py` to reproduce the figures based on the previous `.npy` files.

