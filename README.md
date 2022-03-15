# High-Dimensional Portfolio Selecton with Cardinality Constraints


# Author Contributions Checklist Form

## Data

### Abstract

The case studies considered in this paper are implemented using three datasets including NYSE, S&P 500 and Russell 2000. These three datasets are commercial, proprietary datasets collected from the Center for Research in Security Prices (CRSP) database access through the Wharton Research Data Service (WRDS; www.wrds.upenn.edu).

We collect daily S&P 500 data from 2010 to 2020,  Russell 2000 data from 2004 to 2020 with mergers and acquisitions and reblancing taken into consideration. We collect monthly NYSE data from 2016 to 2018 including 3680 stocks for a simulation purpose.

### Availability

Due to the contract signed at the time of purchase, all three datasets unfortunately cannot be made publicly available. These datasets are proprietary, purchased through WRDS and Siblis Research, Inc. The contract heavily restricts even characteristics of the data (for example, information on stock prices that appear in the datasets). 

Because of this, we have provided tickers of the stocks used in these three datasets. For verification purposes, we have also provided the complete code to run the experiments and generate the results in their entirety, as this has been where the most interest has been generated from our paper thus far. In addition, we provide a minimal working example to demonstrate the type of results shown in the proprietary case study.

If readers want to get access through data used in this paper, they can obtain data with following steps. For NYSE data, WRDS provides information whether a stock is issued in Nasdaq; for S\&P 500 and Russell 2000 data, the reader needs to purchase historical component list information from Siblis Research or obtain it other sources, and then obtain price data from WRDS based on the lists.


### Description

We provide the lists of stock tickers that we use to obtain data from WRDS in this repository.



## Code

### Abstract

The code includes all functions necessary to run the results found in the paper.

We encourage readers to see the `README.md` file accompanying the submitted data and code for examples of the reproducible results code, as well as full details on the main functions (including arguments and outputs).

On the other hand, we also provide a notebook `Minimal_Working_Example.ipynb` for readers who simply want to know how to use our algorithm on their own data.


### Description

This code is delivered via the files described above.

Python (version 3.6 or later) is required to run the files, and it has only been tested on the Linux and the MacOS platforms.

Python packages to run spo functions:

- numba=0.51.2
- numpy=1.19.5
- pandas=1.1.5
- scikit-learn=0.24.2
- scipy=1.5.3
- tqdm=4.62.3

Python packages to run reproducible code:

- cvxopt=1.2.7
- empyrical=0.5.5
- joblib=1.1.0
- numba=0.51.2
- numpy=1.19.5
- pandas=1.1.5
- scikit-learn=0.24.2
- scipy=1.5.3
- tqdm=4.62.3
- nonlinshrink=0.7


No general hardware is required for this code.


### Additional information (optional)


The directories and files are summarized as follows:

- `spo/`: It contains the package implementing the algorithm.


- `data/`: After putting the raw data to `data/raw/`, one can run the script `data/preprocess.py` to preprocess the raw data. The preprocessed data would appear in the other folders and one can proceed to run different experiments later.
    - `raw/`: It contains all raw data obtained from Wharton CRSP historical stock return database and index historical component information.
    - `nyse/`
    - `sp/`
    - `russell/`

- `experiment/`: It contains Python scripts to run all experiments in the paper. The following three subdirectories correspond to reproducing results of Section 4.2, 4.3, and 4.4 in the paper respectively:

    - `simulation/`
    - `sp/`
    - `russell/`

After execute all scripts, the results would be stored in a new directory called `result/`.

- `summary.py`: A Python script to reproduce all the table in the paper, which prints to standard out.
- `plotting.py`: A Python script to reproduce all the figures in the paper, where the images would be stored in a new directory called `img/`.
- `Minimal_Working_Example.ipynb`: A Jupyter Notebook to show minimal working example for running our algorithm.




## Reproducibility workflow

The following workflow should be followed:

1.	Install and load necessary packages.
2.	Run `data/preprocess.py` to preprocess the data. Only the first session in the script is needed for reproducing results of Section 4.2.
3.	Run the following scripts to generate result files:
    - `experiment/simulation/spo_screening_time.py`
    - `experiment/simulation/spo_screening_ratio.py`
    - `experiment/simulation/spo_screening_dualgap.py`
    - `experiment/simulation/spo_efficient_frontier.py`
    - `experiment/simulation/mv_efficient_frontier.py`
4.	Run the summarizing script `summary.py` and plotting script `plotting.py`. Only the first two sections in the script are needed for reproducing results of Section 4.2.


For results in Section 4.3 and 4.5, we have provided the tickers used in WRDS, so that the readers can obtain stock data easier.
After the readers obtain historical component list information, the results in Section 4.3 and 4.5 can be reproduced following the workflow.

1.	Install and load necessary packages.
2.	Run `data/preprocess.py` to preprocess the data. Only the codes after the first session in the script are needed for reproducing results of Section 4.3 and 4.4.
3.	Run the following scripts to generate result files:
    - `experiment/sp/sp_ew.py`
    - `experiment/sp/sp_gmv.py`
    - `experiment/sp/sp_mv_cv.py`
    - `experiment/sp/sp_spo_cv.py`
    - `experiment/sp/sp_spo_n.py`
    - `experiment/russell/russell_ew.py`
    - `experiment/russell/russell_gmv.py`
    - `experiment/russell/russell_mv_cv.py`
    - `experiment/russell/russell_spo_cv.py`
    - `experiment/russell/russell_spo_n.py`
4.	Run the summarizing script `summary.py` and plotting script `plotting.py`. Only the codes after the second section in the script are needed for reproducing results of Section 4.3 and 4.4.

