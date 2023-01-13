#!/usr/bin/env python

#-------------------------------------------------------
# Summary Statistics and Linkage Disequilibrium

# Helper functions for running HEELS

# Version: 0.0.1
#-------------------------------------------------------

import os, sys, re
import logging, time, traceback
import argparse
from functools import reduce
import pickle
import joblib

import pandas as pd
import numpy as np
import random
import math

from scipy.sparse import linalg, coo_matrix
from scipy.linalg import qr, cho_factor, cho_solve, sqrtm, pinvh
from scipy.linalg import cholesky_banded, cho_solve_banded, eigvals_banded, eig_banded, solve_banded
from scipy.optimize import minimize, Bounds
from pandas_plink import read_plink1_bin, read_grm, read_rel
import sklearn.utils.extmath as skmath

import faulthandler; faulthandler.enable()

def band_format(A, bandwidth = None):
    N = np.shape(A)[0]
    if bandwidth >= N:
        # logging.info("Specified bandwidth is not binding to the current LD size.")
        ab = np.zeros((N,N))
        for i in np.arange(N):
            ab[i,:(N-i)] = np.diag(A,k=i)
    else:
        # start_time = time.time()

        if bandwidth is None:
            logging.info("Bandwidth is not provided - using # of non-zero elements of first row as bandwidth.")
            D = np.count_nonzero(A[0,:])
        else:
            D = bandwidth

        ab = np.zeros((D,N))
        for i in np.arange(D):
            ab[i,:(N-i)] = np.diag(A,k=i)

        # time_elapsed = round(time.time() - start_time, 2)
        # logging.info('Banding time: {T}'.format(T=time_elapsed))
    return ab

def band_lu_format(A, bandwidth = None):
    N = np.shape(A)[0]
    if bandwidth is None:
        logging.info("Bandwidth is not provided - using # of non-zero elements of first row as bandwidth.")
        D = np.count_nonzero(A[0,:])
    else:
        D = bandwidth
    ab = np.zeros((D,N))
    ab[0,:N] = np.diag(A,k=0)
    cd = np.zeros((D-1,N))
    for i in np.arange(1, D):
        ab[i,:(N-i)] = np.diag(A, k=i)
        cd[-i, -(N-i):] = np.diag(A, k=i)
    out = np.vstack((cd, ab))
    return out

def offdiag_indices(A, k):
    rows, cols = np.diag_indices_from(A)
    if k < 0:
        return rows[-k:], cols[:k]
    elif k > 0:
        return rows[:-k], cols[k:]
    else:
        return rows, cols

def unband_lower_tri(ab): # ONLY LOWER TRI
    D,N = np.shape(ab)
    A = np.zeros((N,N))
    for i in np.arange(D): 
        A[offdiag_indices(A, -i)] = ab[i, :(N-i)]
    return A

def unband_format(ab): # fill in symmetric full banded mat
    D,N = np.shape(ab)
    A = np.zeros((N,N))
    for i in np.arange(D): 
        A[offdiag_indices(A, -i)] = ab[i, :(N-i)]
    i_lower = np.tril_indices(N, -1)
    B = A.T.copy()
    B[i_lower] = A[i_lower]
    return B

def dense_offband(A, bandwidth):

    m = np.shape(A)[0]
    B = np.zeros_like(A)
    B[np.triu_indices(m, k = bandwidth)] = A[np.triu_indices(m, k = bandwidth)]
    B[np.tril_indices(m, k = -bandwidth)] = A[np.tril_indices(m, k = -bandwidth)]
    B = (B + np.transpose(B)) / 2
    return B

def dense_band(A, bandwidth):

    m = A.shape[0]
    B = A.copy()
    B[np.triu_indices(m, k = bandwidth)] = 0
    B[np.tril_indices(m, k = -bandwidth)] = 0
    B = (B + np.transpose(B)) / 2
    return(B)

