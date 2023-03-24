# `HEELS`
Heritability Estimation with high Efficiency using LD and Summary Statistics

`HEELS` is a Python-based command line tool that produce accurate and precise local heritability estimates using summary-level statistics (marginal association test statistics along with the empirical (in-sample) LD statistics). 

### Getting started
You can clone the repository with the following command:
```
$ git clone git@github.com:huilisabrina/HEELS.git
$ cd HEELS
```
This should take a few seconds to finish. In order to install the Python dependencies, you will need the [Anaconda](https://www.anaconda.com/products/distribution) Python distribution and package manager. After installing Anaconda, run the following commands to create an environment with HEELS' dependencies:
```
conda env create --file heels.yml
source activate heels
```
To test whether installation is done properly, typing the following should give a description of the software and accepted command-line options.
```
$ python3 ./run_HEELS.py -h
```
If an error occurs, then something has gone wrong during installation.

### Updating `HEELS`
You should keep your local instance of this software up to date with the changes that are made on the github repository. To do that, simply type 
```
$ git pull
```
in the `HEELS` directory. If your local instance is outdated, `git` will retrieve all changes and update the code. Otherwise, you will be told that your local instance is already up-to-date. In case the Python dependencies have changed, you can update the HEELS environment with the following

```
conda env update --file heels.yml
```

### Support
We are happy to answer any questions you may have about using the software. Before [opening an issue](https://github.com/huilisabrina/HEELS/issues), please be sure to read the wiki page and read more about our method via the link below. Please also reference the descriptions about our input flags to understand their proper usage. If your problem persists, **please do the following:** as the next step:

  1. Rerun the specification that is causing the error, being sure to specify `--verbose` to generate a descriptive logfile. 
  2. Attach your log file in the issue. 

You may also contact us via email, although we encourage github issues so others can benefit from your question as well.  

### Citation
If you are using the `HEELS` method or software, please cite [Li, Hui, et al. (2023) Accurate and Efficient Estimation of Local
Heritability using Summary Statistics and LD Matrix, 2023](https://www.biorxiv.org/content/10.1101/2023.02.08.527759v2.abstract). doi: <https://doi.org/10.1101/2023.02.08.527759>. 

### License
This package is available under the MIT license.

### Authors
Hui Li (Biostatistics Department, Harvard T.H. Chan School of Public Health)

Rahul Mazumder (Operations Research Center and Center for Statistics and Data Science, MIT Sloan School of Management)

Xihong Lin (Biostatistics Department and Statistics Department, Harvard University)

### Acknoledgement
We are very grateful to Luke O’Connor, Huwenbo Shi, Alkes Price, Samuel Kou, Ben Neale and Shamil Sunyaev for their helpful discussions and feedback. This work was supported by grants R35-CA197449, 548U19-CA203654, R01-HL163560, U01-HG012064 and U01-HG009088 (X. Lin).
