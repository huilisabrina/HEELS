#!/usr/bin/python

#-------------------------------------------------------
# Directly compute the standardized LD matrix
#-------------------------------------------------------
import os, sys, re
import logging, time, traceback
import argparse
from functools import reduce
import pickle

import pandas as pd
import numpy as np
import numpy.matlib
import random
import math
from pandas_plink import read_plink1_bin, read_grm, read_rel, read_plink

import faulthandler; faulthandler.enable()


def sec_to_str(t):
    '''Convert seconds to days:hours:minutes:seconds'''
    [d, h, m, s, n] = reduce(lambda ll, b : divmod(ll[0], b) + ll[1:], [(t, 1), 60, 60, 24])
    f = ''
    if d > 0:
        f += '{D}d:'.format(D=d)
    if h > 0:
        f += '{H}h:'.format(H=h)
    if m > 0:
        f += '{M}m:'.format(M=m)

    f += '{S}s'.format(S=s)
    return f

# standardize LD (cov --> correlation)
def cov_to_corr(covariance):
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation


## Argument parsers
parser = argparse.ArgumentParser(description="\n Simulation pipelines for multi-phenotypes")

parser.add_argument('--plink_bfile_fp', default=None, type=str, 
    help='File path prefix of the genotypic files of causal variants.')
parser.add_argument('--output_fp', default=None, type=str, 
    help='Output prefix name')
parser.add_argument('--stream-stdout', default=False, action="store_true", help='Stream log information on console in addition to writing to log file.')

if __name__ == '__main__':

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s %(message)s', filename=args.output_fp + '.log', filemode='w', level=logging.INFO, datefmt='%Y/%m/%d %I:%M:%S %p')
    if args.stream_stdout:
        logging.getLogger().addHandler(logging.StreamHandler()) # prints to console
    start_time = time.time()

    try:
        # read in causal variant information
        logging.info("Reading genotypic files in PLINK binary format")
        (bim, fam, bed) = read_plink(args.plink_bfile_fp)
        m = bim.shape[0]
        n = fam.shape[0]

        # standardized genotype
        logging.info("Standardizing genotype matrix")
        geno_mat = bed.compute().T # M x N matrix
        geno_avg = np.nanmean(geno_mat, axis = 0)

        # fill in missing values with the avg
        nanidx = np.where(np.isnan(geno_mat))
        geno_mat[nanidx] = geno_avg[nanidx[1]]
        geno_sd = np.nanstd(geno_mat, axis = 0)

        geno_mat = (geno_mat - geno_avg) / geno_sd
        logging.info("Finish standardizing the genotype matrix")
        LD = np.transpose(geno_mat).dot(geno_mat) / n

        # normalize diag of LD
        LD = cov_to_corr(LD)
        logging.info("Finish standardizing the LD and saving to files")
        
        # save standardized genotype and LD
        np.savez(args.output_fp + "_std_geno", std_geno = geno_mat)
        np.savez(args.output_fp + "_std_LD_corr", LD = LD)

    except Exception as e:
        logging.error(e,exc_info=True)
        logging.info('Analysis terminated from error at {T}'.format(T=time.ctime()))
        time_elapsed = round(time.time() - start_time, 2)
        logging.info('Total time elapsed: {T}'.format(T=sec_to_str(time_elapsed)))

    logging.info('Total time elapsed: {}'.format(sec_to_str(time.time()-start_time)))
