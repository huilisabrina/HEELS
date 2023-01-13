# `HEELS`
Heritability Estimation with high Efficiency using LD and Summary Statistics

`HEELS` is a Python-based command line tool that produce accurate and precise local heritability estimates using summary-level statistics (marginal association test statistics and the in-sample LD statistics). For more details please see Li et al. (2023).

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
To test proper installation, ensure that typing 
```
$ python3 ./run_HEELS.py -h
```
gives a description of the software and accepted command-line options. If an error is thrown then something as gone wrong during the installation process.

### Updating `HEELS`
You should keep your local instance of this software up to date with updates that are made on the github repository. To do that, type 
```
$ git pull
```
in the `HEELS` directory. If your local instance is outdated, `git` will retrieve all changes and update the code. Otherwise, you will be told that your local instance is already up to date. In case the Python dependencies have changed, you can update the HEELS environment with

```
conda env update --file heels.yml
```

### Support
We are happy to answer any questions you may have about using the software. Before [opening an issue](https://github.com/huilisabrina/HEELS/issues), please be sure to read the wiki, description of the method in the papers linked above, and the description of the input flags and their proper usage. If your problem persists, **please do the following:**

  1. Rerun the specification that is causing the error, being sure to specify `--verbose` to generate a descriptive logfile. 
  2. Attach your log file in the issue. 
  
You may also contact us via email, although we encourage github issues so others can benefit from your question as well!    

### License
This project is licensed under GNU GPL v3.

### Authors and citation
Hui Li (Biostatistics Department, Harvard T.H. Chan School of Public Health)
Rahul Mazumder (Operations Research Center and Center for Statistics and Data Science, MIT Sloan School of Management)
Xihong Lin (Biostatistics Department and Statistics Department, Harvard University)

If you are using the `HEELS` method or software, please cite Li, Hui et al. "Accurate and Efficient Estimation of Local
Heritability using Summary Statistics and LD Matrix". *bioRxiv*. (2023).
