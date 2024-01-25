#!/usr/bin/env python

#-------------------------------------------------------
# HEELS package
# Run the HEELS algorithm (w/ full LD or sparse approx)

# Required modules:
# - utils/heels_format
# - utils/heels_utils
# - heels_LDcecomp

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

# import self-defined functions
sys.path.append('/n/holystore01/LABS/xlin/Lab/huili/HEELS/main_func/utils')
sys.path.append('/n/holystore01/LABS/xlin/Lab/huili/HEELS/main_func')

from utils import heels_utils
from utils import heels_format
import heels_LDdecomp

__version__ = '0.0.3'

borderline = "<><><<>><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>"
short_borderline = "<><><<>><><><><><><><><><><><><><><><><><><><><><><><><><><><><>\n"
header ="\n"
header += borderline +"\n"
header += "<>\n"
header += "<> HEELS: Heritability Estimator with high Efficiency using LD and Summary Statistics \n"
header += "<> Version: {}\n".format(str(__version__))
header += "<> (C) 2022 Hui Li, Rahul Mazumder and Xihong Lin\n"
header += "<> Harvard University Department of Biostatistics\n"
header += "<> MIT Sloan School of Management, Operations Research Center and Center for Statistics and Data Science\n"
header += "<> MIT License Copyright (c) 2023 Hui Li \n"
header += borderline + "\n"
header += "<> Note:  It is recommended to run your own QC on the input before using this program. \n"
header += "<> Software-related correspondence: hui_li@g.harvard.edu \n"
header += borderline +"\n"
header += "\n\n"

pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 800)
pd.set_option('display.precision', 12)
pd.set_option('max_colwidth', 800)
pd.set_option('colheader_justify', 'left')

np.set_printoptions(linewidth=800)
np.set_printoptions(precision=3)

# Functions for subrountine
def read_sparse_LD(args, block_path, block_number = None):

    '''
    Read in LD matrix in its sparse representation.
    NOT to be used for reading full LD matrix.

    Parameters:
    -----------
    block_path: file path to the LD matrix or matrices
    block_number: index of the LD block, if multiple blocks are specified


    Returns:
    ---------
    LD_dict: dictionary of the LD information. Contents vary depending on the method
    m_ref: number of reference variants


    '''
    if block_number is None:
        logging.info("Reading in sparse representation of LD")
    else:
        logging.info("Reading in sparse representation of LD block {}".format(block_number))

    if args.LD_approx_mode is None or args.LD_approx_method is None:
        raise ValueError("Requires specifying --LD_approx_mode and --LD_approx_method when --LD_approx_path is not None")
    else:
        logging.info("Sparse decomposition based on the method: {}".format(args.LD_approx_method))
        
        # read in B,R if the mode is auto
        if args.LD_approx_mode == "auto":
            args.block_B = int(np.loadtxt(block_path + ".opt_b"))
            args.block_R = int(np.loadtxt(block_path + ".opt_r"))
            logging.info("Adopted optimal (B,R) values are: ({},{})".format(block_B, args.block_R))

        # unified reading of all alternative methods
        if args.LD_approx_method == "Band_only":
            decomp_objs = np.load(block_path + ".npz")
            LD_b = decomp_objs['LD_b']
            args.block_B = np.count_nonzero(LD_b[0,:])
            logging.info("Bandwidth of the banded component used in the LD approx: {}".format(args.block_B))

            # compute norm of the err
            # F_norm = np.linalg.norm(LD - LD_b)

        elif args.LD_approx_method == "LR_only":
            decomp_objs = np.load(block_path + ".npz")
            eigval = decomp_objs['w']
            eigvec = decomp_objs['v']

            # reconstruct LD based on the truncation
            LD_lr = eigvec.dot(np.diag(eigval).dot(np.transpose(eigvec)))
            args.block_R = eigval.shape[0]
            logging.info("Number of low-rank factors used in the LD approx: {}".format(args.block_R))

            # compute norm of the err
            # F_norm = np.linalg.norm(LD - LD_lr)

        else:
            decomp_objs = np.load(block_path + ".npz")
            eigval = decomp_objs['w']
            eigvec = decomp_objs['v']
            LD_b = decomp_objs['LD_b']
            args.block_B = np.count_nonzero(LD_b[0,:])
            logging.info("Bandwidth of the banded component used in the LD approx: {}".format(args.block_B))

            # reconstruct LD based on the truncation
            LD_lr = eigvec.dot(np.diag(eigval).dot(np.transpose(eigvec)))
            args.block_R = eigval.shape[0]
            logging.info("Number of low-rank factors used in the LD approx: {}".format(args.block_R))

            # compute norm of the err
            # F_norm = np.linalg.norm(LD - LD_lr - LD_b)

        # additional adjustment for the sparse representation part
        if args.LD_approx_method != "Band_only":
            # logging.info("(Check) Smallest eigenvalues used for the LR component: ")
            # logging.info(eigval[-10:])

            #=======================
            # Inducing sparsity
            #=======================
            # threshold eigenvectors
            if args.LD_approx_thres_eigvec > 0:
                logging.info("Thresholding low-rank matrix at {}".format(args.LD_approx_thres_eigvec))
                # logging.info("Number of regularized elements: {}".format(np.sum(np.abs(eigvec) < float(args.LD_approx_thres_eigvec))))
                logging.info("Sparsity of the eigenvectors for the LR component: {}".format(np.sum(np.abs(eigvec) < float(args.LD_approx_thres_eigvec)) / (m*args.block_R)))
                eigvec[np.abs(eigvec) < float(args.LD_approx_thres_eigvec)] = 0


    # organize the readings of the input data into LD_dict and m_ref
    if args.LD_approx_method == "LR_only":
        LD_dict = dict({'eigvec': eigvec, 'eigval': eigval})
        m_ref = eigvec.shape[0]
    elif args.LD_approx_method == "Band_only":
        LD_dict = dict({'LD_banded': heels_format.band_format(LD_b, args.block_B)})
        m_ref = LD_b.shape[0]
    elif args.LD_approx_method is not None:
        LD_dict = dict({'LD_banded': heels_format.band_format(LD_b, args.block_B), 'eigvec': eigvec, 'eigval': eigval})
        m_ref = eigvec.shape[0]

    # else: # NO APPROX or ORIG
        # LD_dict = dict({'LD_banded': heels_format.band_format(LD, LD.shape[0])})

    return LD_dict, m_ref

def read_sumstats(fp):
    '''
    Read in the summary statistics. Check if Z and n columns exist
    Useful for reading of multiple files for different LD blocks.

    
    Parameters:
    ------------
    fp: file path to the summary statistics with the Z-scores
    
    Returns:
    ---------
    df_ss: summary statistics
    n_avg: average sample size

    '''
    df_ss = pd.read_csv(fp, delim_whitespace=True, index_col=None)

    assert 'Z' in list(df_ss.columns), "The summstats is missing a Z column!"
    assert 'n' in list(df_ss.columns), "The sumstats is missing the n column!"

    # average N from sumstats for the current block
    n_avg = np.mean(df_ss['n'].values)

    return df_ss, n_avg

def align_sumstats_LD(args, block_bim_path, df_ss, block_LD_dict, block_m, block_number = None):

    '''
    Aligning the summary statistics with the LD matrix.

    
    Parameters:
    ------------
    block_bim_path: file path to the bim file of the LD block
    df_ss: a dataframe of summary statistics with the Z-scores
    block_LD_dict: LD dictionary of the LD block
    block_m: block size
    block_number: block index


    Returns:
    ---------
    block_LD_dict: LD information of the overlap between sumstats and LD
    df_both: summary statistics of the overlap between sumstats and LD

    '''

    if block_number is None:
        logging.info("Aligning sumstats with LD info and selecting overlapped markers only")
    else:
        logging.info("Aligning sumstats with LD info for block {}".format(block_number))

    # read in LD-SNP info file
    ld_snp_fp = block_bim_path
    df_bim = pd.read_csv(ld_snp_fp, delim_whitespace=True, index_col=None, names=['CHR','SNP','pos','BP','A1','A2'])

    assert block_m == df_bim.shape[0], "Dim mismatch between LD-SNP info file and the LD input!"

    # overlapped num of markers (TO DO: acccommodate diff colnames)
    df_merge = pd.merge(df_bim[["CHR", "BP", "SNP"]], df_ss, left_on = ["CHR", "BP", "SNP"], right_on = ["CHR", "BP", "Predictor"], how = 'left', indicator = "merge_flag")
    df_both = df_merge[df_merge['merge_flag'] == "both"]
    m = df_both.shape[0] # number of overlapped markers between sumstats and LD
    n = np.mean(df_both['n'].values) # this overwrites the input N value

    # select overlapped SNPs for LD
    LDsnp_index = (df_merge['merge_flag'] == "both")
    overlap_index = np.arange(block_m)[LDsnp_index]

    # summarize SNP counts after alignment
    logging.info("Number of SNPs in the LD: {block_m}".format(block_m = block_m))
    logging.info("Number of SNPs overlapped between sumstats and LD {m}".format(m = m))
    
    # sample size adjustment factor
    nm_adj = (n/m) / (n_ref/block_m)

    # SNP alignment between sumstats and LD (depending on the method, required adjustments differ)

    # Sparse representation of LD is used
    if args.LD_approx_method == "LR_only":
        block_LD_dict['eigvec'] = block_LD_dict['eigvec'][overlap_index, :] * math.sqrt(nm_adj)
    elif args.LD_approx_method == "Band_only":
        banded_filter = np.arange(block_m)[overlap_index][:block_LD_dict['LD_banded'].shape[0]]
        block_LD_dict['LD_banded'] = block_LD_dict['LD_banded'][banded_filter[:, np.newaxis], overlap_index] * nm_adj
    elif args.LD_approx_method is not None:
        banded_filter = np.arange(block_m)[overlap_index][:block_LD_dict['LD_banded'].shape[0]]
        block_LD_dict['LD_banded'] = block_LD_dict['LD_banded'][banded_filter[:, np.newaxis], overlap_index] * nm_adj
        block_LD_dict['eigvec'] = block_LD_dict['eigvec'][overlap_index, :] * math.sqrt(nm_adj)
    else:
        block_LD_dict['LD_banded'] = block_LD_dict['LD_banded'][overlap_index[:, np.newaxis], overlap_index] * nm_adj

    return block_LD_dict, df_both

def chol_fac_banded(W_t_tilde, m):
    '''Using Cholesky factorization to invert a PSD matrix. '''
    chol = cholesky_banded(W_t_tilde, lower = True)
    precond = cho_solve_banded((chol, True), np.eye(m))

    return precond

def HEELS_lr_iter(lam, m, Z_m, eigvec, eigval, YtY=1, sigma_e=None):

    '''
    HEELS updating subroutine if the LD is approximated by a low-rank matrix only.

    Notes: 
    -- When sigma_e is left unspecified, this function can be used for block-wise BLUP update. When sigma_e is specified, this function is used to update the BLUP at each iteration. 

    '''

    # use Woodbury identity to solve W^{-1}
    Ax = Z_m / lam
    lowrank = np.linalg.inv(np.diag(1 / eigval) + np.transpose(eigvec).dot(eigvec / lam))
    Abbax = eigvec.dot(lowrank.dot(np.transpose(eigvec).dot(Ax))) / lam
    BLUP_t = np.asarray(Ax - Abbax)

    # use the dot product trick to obtain trace(W^{-1})
    lowrank_rt = sqrtm(lowrank) # OPTIMIZE: combine it with lowrank - use eigdecomp once to avoid inversion twice
    Wuv = eigvec.dot(lowrank_rt) / lam
    trace = m / lam - np.sum(np.multiply(Wuv, Wuv))

    # compute the (approx) likelihood when LD is approximated with a low-rank matrix
    if sigma_e is not None:
        Z_ll = (-1)*0.5*(n*np.log(sigma_e) + np.sum(np.log(eigval)) - m*np.log(lam) + ((YtY - np.sum(np.multiply(BLUP_t, np.asarray(Z_m))))/sigma_e))
    else:
        Z_ll = None

    return BLUP_t, trace, Z_ll

def HEELS_band_iter(lam, m, Z_m, LD_b, YtY=1, sigma_e=None):

    '''
    HEELS updating subroutine if the LD is approximated by a banded matrix.

    Notes: 
    -- Banded matrix is NOT guaranteed to be PSD.
    -- When sigma_e is left unspecified, this function can be used for block-wise BLUP update. When sigma_e is specified, this function is used to update the BLUP at each iteration. 

    '''

    b = int(LD_b.shape[0])
    W_t = LD_b.copy()
    W_t[0,:] = W_t[0,:] + lam

    # safest but slowest option
    # inv = np.linalg.inv(heels_format.unband_format(W_t))
    # BLUP_t = inv.dot(Z_m)

    # use the banded structure
    W_lu = heels_format.band_lu_format(heels_format.unband_format(W_t), b)
    BLUP_t = solve_banded((b-1, b-1), W_lu, Z_m)
    
    # use banded structure to get the inverse (may be parallelizable)
    inv = solve_banded((b-1, b-1), W_lu, np.eye(m))
    trace = np.trace(inv)

    # compute the (approx) likelihood when LD is approximated with a banded matrix
    if sigma_e is not None:
        # compute eigenvalues of the banded part of W
        w_approx = eigvals_banded(W_t, lower=True)
        eigvals_b = w_approx[::-1]

        # calculate log-likelihood from sumstats + LD
        Z_ll = (-1)*0.5*(n*np.log(sigma_e) + np.sum(np.log(eigvals_b)) - m*np.log(lam) + ((YtY - np.sum(np.multiply(BLUP_t, np.asarray(Z_m))))/sigma_e))
    else:
        Z_ll = None

    return BLUP_t, trace, Z_ll

def HEELS_band_lr_iter(lam, m, Z_m, LD_b, eigvec, eigval, LD_approx_method, YtY=1, sigma_e=None):

    '''
    HEELS updating subroutine if the LD is approximated by Banded + LR components.

    Notes:
    -- if "seq_band_lr" or "seq_lr_band" is used, the banded component of the approximation is NOT guaranteed to be PSD. Hence, a general version of the inverse function is used; 
    -- if "PSD_band_lr" or "joint" is used, we can exploit the PSD of the banded component to compute its inverse.
    -- When sigma_e is left unspecified, this function can be used for block-wise BLUP update. When sigma_e is specified, this function is used to update the BLUP at each iteration. 

    '''

    b = int(LD_b.shape[0])
    W_t = LD_b.copy()
    W_t[0,:] = W_t[0,:] + lam

    if LD_approx_method == "seq_band_lr" or LD_approx_method == "seq_lr_band" or LD_approx_method == "seq":
        W_lu = heels_format.band_lu_format(heels_format.unband_format(W_t), b)
        # Ax = solve_banded((b-1, b-1), W_lu, Z_m)
        inv = solve_banded((b-1, b-1), W_lu, np.eye(m))

    else: # use Chol directly if W_b is known to be PSD
        inv = chol_fac_banded(W_t, m)

        # if LD_b is given in LL' form, we may use rank-1 update to L and find the chol factor of W_t. check out Alg.3.1 of the CS paper. 

    Ax = inv.dot(Z_m)
    lowrank = np.linalg.inv(np.diag(1 / eigval) + np.transpose(eigvec).dot(inv.dot(eigvec)))
    Abbax = inv.dot(eigvec.dot(lowrank.dot(np.transpose(eigvec).dot(Ax))))
    BLUP_t = Ax - Abbax

    # solve (tr(W^{-1}))
    lowrank_rt = sqrtm(lowrank)
    Wuv = inv.dot(eigvec.dot(lowrank_rt))
    trace = np.trace(inv) - np.sum(np.diag(np.transpose(Wuv).dot(Wuv)))

    # compute the likelihood using the LD approx
    if sigma_e is not None:
        # compute eigenvalues of the banded part of W
        w_approx = eigvals_banded(W_t, lower=True)
        eigvals_b = w_approx[::-1]
        r = eigval.shape[0]

        # calculate log-likelihood from sumstats + LD
        Z_ll = (-1)*0.5*(n*np.log(sigma_e) + np.sum(np.log(eigvals_b[:r] + eigval)) + np.sum(np.log(eigvals_b[r+1:])) - m*np.log(lam) + ((YtY - np.sum(np.multiply(BLUP_t, np.asarray(Z_m))))/sigma_e))

    else:
        Z_ll = None
    
    return BLUP_t, trace, Z_ll

def HEELS_iter(lam, m, n, Z_m, LD_0, YtY=1, sigma_e=None):

    '''
    HEELS updating subroutine if LD is given in its full dense format.

    When sigma_e is left unspecified, this function can be used for block-wise BLUP update. 
    When sigma_e is specified, this function is used to update the BLUP at each iteration. 

    '''

    # solve W_t^{-1}Z:
    W_t = LD_0.copy()
    W_t[0,:] = W_t[0,:] + lam

    # solve W^{-1}Z:
    # since LD_0 naturally should be PSD
    chol = cholesky_banded(W_t, lower = True)
    BLUP_t = cho_solve_banded((chol, True), Z_m)

    # solve (tr(W^{-1}))
    inv = cho_solve_banded((chol, True), np.eye(m))

    # DEBUG: solve inverse directly
    # LD_fullmat = heels_format.unband_format(W_t)
    # inv = np.linalg.inv(LD_fullmat)
    
    trace = np.trace(inv)

    if sigma_e is not None:
        # calculate log-likelihood from sumstats + LD
        Z_ll = (-1)*0.5*(n*np.log(sigma_e) + 2*np.sum(np.log(chol[0,:])) - m*np.log(lam) + ((YtY - np.sum(np.multiply(BLUP_t, np.asarray(Z_m))))/sigma_e))

        # the following does not work due to numerical overflow
        # det_W = np.prod(np.square(chol[0,:]) / lam)
        # Z_ll = (-1)*0.5*(n*np.log(sigma_e) + np.log(det_W) + ((YtY - np.sum(np.multiply(BLUP_t, Z_m)))/sigma_e))

        # chol_decile = np.percentile(chol[0,:], np.arange(0, 100, 10))
        # logging.info(chol_decile)

    else:
        Z_ll = None
    
    return BLUP_t, trace, Z_ll

def HEELS_band_lr(args, Z_m, LD_b, eigvec, eigval, sigma_g_0, sigma_e_0, m, n, YtY, tol=1e-3, maxIter=100):
    
    '''
    Wrapper function for estimating the variance componnets. 

    Used for CV functions in LD decomposition. 

    '''
    # initialize algorithm
    sigma_g = sigma_g_0
    sigma_e = sigma_e_0
    sigma_g_list = [sigma_g]
    sigma_e_list = [sigma_e]
    diff_g = 100
    diff_e = 100

    # timing of the iterative procedure
    start_time = time.time()
    logging.info('HEELS procedure started at {T}'.format(T=time.ctime()))
    i = 0

    while ((abs(diff_g) > tol or abs(diff_e) > tol) and (i < maxIter)):
        logging.info("Iteration: {}".format(i))

        sigma_g_prev = sigma_g 
        sigma_e_prev = sigma_e
        lam = sigma_e / sigma_g

        # update BLUP estimates
        BLUP_t, trace, _ = HEELS_band_lr_iter(lam, m, Z_m, LD_b, eigvec, eigval, args.LD_approx_method)

        # update variance components
        logging.info("Updating variance components: ")
        sigma_g = 1/m*(np.matmul(np.transpose(BLUP_t), BLUP_t).item(0) + sigma_e*trace)

        if args.approx_YtY_bs:
            sample_varY = n*np.mean((args.beta**2 + args.se**2)*2*np.multiply(args.freq, 1-args.freq))
            logging.info("Using computed sample variance of {} from sumstats".format(sample_varY))
            sigma_e = sample_varY - np.matmul(np.transpose(Z_m), BLUP_t).item(0)/n
        elif YtY is not None: # if y'y is known
            YtY = round(float(YtY), 16)
            logging.info("Using sample variance of {} to update sigma_e^2".format(YtY))
            sigma_e = YtY - 1/n*(np.matmul(np.transpose(Z_m), BLUP_t).item(0))
        else: # if y'y is left unspecified
            YtY = sigma_g + sigma_e
            sigma_e = YtY - 1/n*(np.matmul(np.transpose(Z_m), BLUP_t).item(0))
        
        if args.constrain_sigma:
            logging.info("Rescaling updated values s.t. sigma_g^2 + sigma_e^2 = 1")
            sigma_tot = sigma_g + sigma_e
            sigma_g = np.real(sigma_g / sigma_tot)
            sigma_e = np.real(sigma_e / sigma_tot)

        # ==============
        # record results
        # ==============
        diff_g = sigma_g - sigma_g_prev
        diff_e = sigma_e - sigma_e_prev
        logging.info("Current sigma_g: {} \n".format(sigma_g))
        logging.info("Current sigma_e: {} \n".format(sigma_e))
        logging.info("Difference in sigma_g: {} \n".format(diff_g))
        logging.info("Difference in sigma_e: {} \n".format(diff_e))
        sigma_g_list.append(sigma_g)
        sigma_e_list.append(sigma_e)
        i = i + 1

    time_elapsed = round(time.time() - start_time, 2)
    logging.info('Time elapsed in seconds: {T}'.format(T=time_elapsed))
    logging.info('Time elapsed for iterative procedure: {T}'.format(T=heels_utils.sec_to_str(time_elapsed)))
    logging.info('Number of iterations: {I}'.format(I=i))

    return sigma_g, sigma_e

def HEELS_band(args, Z_m, LD_b, sigma_g_0, sigma_e_0, m, n, YtY, tol=1e-3, maxIter=100):
    
    '''
    Wrapper function for estimating the variance componnets. 

    Used for CV functions in LD decomposition. 

    '''
    # initialize algorithm
    sigma_g = sigma_g_0
    sigma_e = sigma_e_0
    sigma_g_list = [sigma_g]
    sigma_e_list = [sigma_e]
    diff_g = 100
    diff_e = 100

    # timing of the iterative procedure
    start_time = time.time()
    logging.info('HEELS procedure started at {T}'.format(T=time.ctime()))
    i = 0

    while ((abs(diff_g) > tol or abs(diff_e) > tol) and (i < maxIter)):
        logging.info("Iteration: {}".format(i))

        sigma_g_prev = sigma_g 
        sigma_e_prev = sigma_e
        lam = sigma_e / sigma_g

        # update BLUP estimates
        BLUP_t, trace, _ = HEELS_band_iter(lam, m, Z_m, LD_b)

        # update variance components
        logging.info("Updating variance components: ")
        sigma_g = 1/m*(np.matmul(np.transpose(BLUP_t), BLUP_t).item(0) + sigma_e*trace)

        if args.approx_YtY_bs:
            logging.info("Using beta and SE to approximate sample variance")
            sample_varY = np.mean(args.beta**2 + n*args.se**2)
            sigma_e = sample_varY - 1/n*(np.matmul(np.transpose(Z_m), BLUP_t).item(0))
        elif YtY is not None: # if y'y is known
            YtY = round(float(YtY), 16)
            logging.info("Using sample variance of {} to update sigma_e^2".format(YtY))
            sigma_e = YtY - 1/n*(np.matmul(np.transpose(Z_m), BLUP_t).item(0))
        else:
            sigma_e = 1/n*(n - np.matmul(np.transpose(Z_m), BLUP_t).item(0))
        
        if args.constrain_sigma:
            logging.info("Rescaling updated values s.t. sigma_g^2 + sigma_e^2 = 1")
            sigma_tot = sigma_g + sigma_e
            sigma_g = np.real(sigma_g / sigma_tot)
            sigma_e = np.real(sigma_e / sigma_tot)

        # ==============
        # record results
        # ==============
        diff_g = sigma_g - sigma_g_prev
        diff_e = sigma_e - sigma_e_prev
        logging.info("Current sigma_g: {} \n".format(sigma_g))
        logging.info("Current sigma_e: {} \n".format(sigma_e))
        logging.info("Difference in sigma_g: {} \n".format(diff_g))
        logging.info("Difference in sigma_e: {} \n".format(diff_e))
        sigma_g_list.append(sigma_g)
        sigma_e_list.append(sigma_e)
        i = i + 1

    time_elapsed = round(time.time() - start_time, 2)
    logging.info('Time elapsed in seconds: {T}'.format(T=time_elapsed))
    logging.info('Time elapsed for iterative procedure: {T}'.format(T=heels_utils.sec_to_str(time_elapsed)))
    logging.info('Number of iterations: {I}'.format(I=i))

    return sigma_g, sigma_e

# HEELS main
def run_HEELS(args, Z_m, LD_dict, sigma_g_0, sigma_e_0, m, n, YtY=None, update_sigma_g="Seq", tol=1e-3, maxIter=100):

    # initialize algorithm
    sigma_g = sigma_g_0
    sigma_e = sigma_e_0
    sigma_g_list = [sigma_g]
    sigma_e_list = [sigma_e]
    diff_g = 100
    diff_e = 100

    # timing of the iterative procedure
    start_time = time.time()
    logging.info('HEELS procedure started at {T}'.format(T=time.ctime()))
    i = 0
    Z_ll_path = [-1000, -500]
    Y_ll_path = []

    # stopping criteria: 1) log-likelihood path; 2) number of iterations
    while (abs(Z_ll_path[-2] - Z_ll_path[-1]) > tol and (i < maxIter)):
        logging.info("Iteration: {}".format(i))

        sigma_g_prev = sigma_g 
        sigma_e_prev = sigma_e
        lam = sigma_e / sigma_g

        if YtY is None:
            # logging.info("Since YtY is not specified, use the sum of sigma_g and sigma_e")
            YtY = sigma_g + sigma_e

        # update the BLUP and compute trace
        if args.LD_approx_method == "LR_only":
            BLUP_t, trace, Z_ll = HEELS_lr_iter(lam, m, Z_m, LD_dict['eigvec'], LD_dict['eigval'], YtY, sigma_e)
        elif args.LD_approx_method == "Band_only":
            BLUP_t, trace, Z_ll = HEELS_band_iter(lam, m, Z_m, LD_dict['LD_banded'], YtY, sigma_e)
        elif args.LD_approx_method is not None:
            BLUP_t, trace, Z_ll = HEELS_band_lr_iter(lam, m, Z_m, LD_dict['LD_banded'], LD_dict['eigvec'], LD_dict['eigval'], args.LD_approx_method, YtY, sigma_e)
        else:
            BLUP_t, trace, Z_ll = HEELS_iter(lam, m, n, Z_m, LD_dict['LD_banded'], YtY, sigma_e)

            if args.check_Y_ll:
                # read in raw genotype
                std_geno = np.load(args.std_geno_fp)['std_geno']
                X = std_geno / math.sqrt(m)
                n = int(X.shape[0])
                # read in raw phenotypes
                df_pheno = pd.read_csv(args.pheno_mat_fp, delim_whitespace=True, index_col=None)
                pheno_mat = df_pheno.to_numpy()
                # remove FID and IID
                pheno_mat = pheno_mat[:, 2:] 
                # read in the specific Y vector
                Y = pheno_mat[:, int(args.pheno_index)]

                # compute the likelihood
                GRM = np.matmul(X, np.transpose(X))
                G_t = heels_format.band_format(GRM, n)
                V_t = G_t.copy()*sigma_g
                V_t[0,:] = V_t[0,:] + sigma_e
                V_chol = cholesky_banded(V_t, lower = True)
                det_V = np.prod(np.square(V_chol[0,:]))
                inv_V = cho_solve_banded((V_chol, True), np.eye(n))
                Y_ll = -0.5*(np.log(det_V) + Y.T.dot(inv_V.dot(Y)))
                Y_ll_path.append(Y_ll)

        Z_ll_path.append(Z_ll)
        
        # use the alternative updating function for sigma_g
        if update_sigma_g == "noSeq": 
            sigma_g = np.matmul(np.transpose(BLUP_t), BLUP_t).item(0) / (m - trace)
        else:
            sigma_g = 1/m*(np.matmul(np.transpose(BLUP_t), BLUP_t).item(0) + sigma_e*trace)

        # use the summary statistics to infer sample variance
        if args.approx_YtY_bs:
            sample_varY = np.mean(args.beta**2 + n*args.se**2)
            sigma_e = sample_varY - 1/n*(np.matmul(np.transpose(Z_m), BLUP_t).item(0))
            logging.info("Using beta and SE to approximate sample variance {}".format(sample_varY))
        else:
            YtY = round(float(YtY), 16)
            logging.info("Using sample variance of {} to update sigma_e^2".format(YtY))
            # logging.info(type(Z_m))
            # logging.info(type(BLUP_t))
            sigma_e = YtY - 1/n*(np.sum(np.multiply(np.asarray(Z_m), BLUP_t)))

        # applying constraints on sigma_g and sigma_e
        if args.constrain_sigma:
            logging.info("Rescaling updated values s.t. sigma_g^2 + sigma_e^2 = 1")
            sigma_tot = sigma_g + sigma_e
            sigma_g = np.real(sigma_g / sigma_tot)
            sigma_e = np.real(sigma_e / sigma_tot)

        # record results
        diff_g = sigma_g - sigma_g_prev
        diff_e = sigma_e - sigma_e_prev
        logging.info("Current sigma_g: {} \n".format(sigma_g))
        logging.info("Current sigma_e: {} \n".format(sigma_e))
        logging.info("Difference in sigma_g: {} \n".format(diff_g))
        logging.info("Difference in sigma_e: {} \n".format(diff_e))
        logging.info("Current Z-logL value: {} \n".format(Z_ll_path[-1]))
        # if args.check_Y_ll:
        #     logging.info("Current Y-logL value: {} \n".format(Y_ll_path[-1]))
        # logging.info("Delta logL: {} \n".format(Z_ll_path[-1] - Z_ll_path[-2]))
        # sigma_g_list.append(sigma_g)
        # sigma_e_list.append(sigma_e)
        i = i + 1

    time_elapsed = round(time.time() - start_time, 2)
    logging.info('Time elapsed in seconds: {T}'.format(T=time_elapsed))
    logging.info('Time elapsed for iterative procedure: {T}'.format(T=heels_utils.sec_to_str(time_elapsed)))
    logging.info('Number of iterations: {I}'.format(I=i))

    return sigma_g, sigma_e, i, sigma_g_list, sigma_e_list


# HEELS Block-wise HEELS
def run_HEELS_block(args, Z_m_list, LD_dict_list, sigma_g_0, sigma_e_0, m_list, n, YtY=None, update_sigma_g="Seq", tol=1e-3, maxIter=100):

    # number of blocks
    K = len(LD_dict_list)

    # check dimension of Z and LD
    check_Z = np.sum([len(Z_m_list[k]) != m_list[k] for k in range(K)])
    assert check_Z == 0, "There are {} blocks with misaligned Z_m and SNP counts".format(check_Z)

    # initialize algorithm
    sigma_g = sigma_g_0
    sigma_e = sigma_e_0
    sigma_g_list = [sigma_g]
    sigma_e_list = [sigma_e]
    diff_g = 100
    diff_e = 100

    # timing of the iterative procedure
    start_time = time.time()
    logging.info('HEELS procedure started at {T}'.format(T=time.ctime()))
    i = 0

    # block-specific handling
    m = np.sum(m_list)
    BLUP_t = np.zeros(shape=(m,))
    traces = np.zeros(shape=(K,))
    Z_index = np.insert(np.cumsum(np.asarray(m_list)), 0, 0) # index of block boundaries
    logging.info(Z_m_list[0].shape)
    Z_m = np.vstack(Z_m_list)
    logging.info(Z_m.shape)

    while ((abs(diff_g) > tol or abs(diff_e) > tol) and (i < maxIter)):
        logging.info("Iteration: {}".format(i))

        sigma_g_prev = sigma_g 
        sigma_e_prev = sigma_e
        lam = sigma_e / sigma_g

        if YtY is None: # currently disabled
            logging.info("Since YtY is not specified, use the sum of sigma_g and sigma_e")
            YtY = sigma_g + sigma_e

        # update BLUP (can be potentially parallelized)
        logging.info("Starting block-wise BLUP updates")
        for k in range(K):
            BLUP_t[Z_index[k]:Z_index[k+1]], traces[k] = block_BLUP_update(args, lam, m_list[k], n, Z_m_list[k], LD_dict_list[k])

        # logging.info("BLUP estimates: ")
        # logging.info(BLUP_t)

        logging.info("Aggregating across blocks for variance component updates")
        # aggregate block results
        trace = np.sum(traces)

        logging.info("Total number of markers: {}".format(m))
        logging.info("Trace term: {}".format(trace))

        # use the alternative updating function for sigma_g
        if update_sigma_g == "noSeq": 
            sigma_g = np.matmul(np.transpose(BLUP_t), BLUP_t).item(0) / (m - trace)
        else:
            sigma_g = 1/m*(np.matmul(np.transpose(BLUP_t), BLUP_t).item(0) + sigma_e*trace)

        # use the summary statistics to infer sample variance
        if args.approx_YtY_bs:
            sample_varY = np.mean(args.beta**2 + n*args.se**2)
            sigma_e = sample_varY - 1/n*(np.matmul(np.transpose(Z_m), BLUP_t).item(0))
            logging.info("Using beta and SE to approximate sample variance {}".format(sample_varY))
        else:
            YtY = round(float(YtY), 16)
            logging.info("Using sample variance of {} to update sigma_e^2".format(YtY))
            sigma_e = YtY - 1/n*(np.matmul(np.transpose(Z_m), BLUP_t).item(0))

        # applying constraints on sigma_g and sigma_e
        if args.constrain_sigma:
            logging.info("Rescaling updated values s.t. sigma_g^2 + sigma_e^2 = 1")
            sigma_tot = sigma_g + sigma_e
            sigma_g = np.real(sigma_g / sigma_tot)
            sigma_e = np.real(sigma_e / sigma_tot)

        # record results
        diff_g = sigma_g - sigma_g_prev
        diff_e = sigma_e - sigma_e_prev
        logging.info("Current sigma_g: {} \n".format(sigma_g))
        logging.info("Current sigma_e: {} \n".format(sigma_e))
        logging.info("Difference in sigma_g: {} \n".format(diff_g))
        logging.info("Difference in sigma_e: {} \n".format(diff_e))
        # sigma_g_list.append(sigma_g)
        # sigma_e_list.append(sigma_e)
        i = i + 1

    time_elapsed = round(time.time() - start_time, 2)
    logging.info('Time elapsed in seconds: {T}'.format(T=time_elapsed))
    logging.info('Time elapsed for iterative procedure: {T}'.format(T=heels_utils.sec_to_str(time_elapsed)))
    logging.info('Number of iterations: {I}'.format(I=i))

    return sigma_g, sigma_e, i, sigma_g_list, sigma_e_list
    
def block_BLUP_update(args, lam, m_sub, n, Z_sub, LD_dict):
    # wrapper for block-specific evaluation of BLUP and trace
    if args.LD_approx_method == "LR_only":
        BLUP_t, trace, _ = HEELS_lr_iter(lam, m_sub, Z_sub, LD_dict['eigvec'], LD_dict['eigval'])
    elif args.LD_approx_method == "Band_only":
        BLUP_t, trace, _ = HEELS_band_iter(lam, m_sub, Z_sub, LD_dict['LD_banded'])
    elif args.LD_approx_method is not None:
        BLUP_t, trace, _ = HEELS_band_lr_iter(lam, m_sub, Z_sub, LD_dict['LD_banded'], LD_dict['eigvec'], LD_dict['eigval'], args.LD_approx_method)
    else:
        BLUP_t, trace, _ = HEELS_iter(lam, m_sub, n, Z_sub, LD_dict['LD_banded'], YtY = args.YtY) # Z_ll kept for heritage

    return BLUP_t.reshape(-1,), trace

def block_BLUP_update_util(lam, m_sub, Z_sub, LD_dict):
    # TO TEST: enable other LD approx strategies
    # currently, only full exact LD is allowed
    BLUP_t, trace, _ = HEELS_iter(lam, m_sub, Z_sub, LD_dict['LD_banded'])

    return BLUP_t.reshape(-1,), trace

def block_sigma_update(args, Z_m, BLUP, trace, sigma_g, sigma_e, m, n):

    # update variance components
    sigma_g = 1/m*(np.matmul(np.transpose(BLUP), BLUP).item(0) + sigma_e*trace)
    sigma_e = 1/n*(n - np.matmul(np.transpose(Z_m), BLUP).item(0))

    return sigma_g, sigma_e

# =================
# DEFINE ARGS 
# =================
## Argument parsers
parser = argparse.ArgumentParser(description="\n Heritability Estimation with high Efficiency using LD and Summary Statistics (HEELS)")

## input and output file paths
IOfile = parser.add_argument_group(title="Input and output options")
IOfile.add_argument('--output_fp', default=None, type=str, 
    help='File path prefix of the output files.')
IOfile.add_argument('--est_fp', default=None, type=str, 
    help='File path prefix of the existing estimates. Used for cases where only the variance estimation is necessary.')
IOfile.add_argument('--sumstats_fp', default=None, type=str, 
    help='File path of the GWAS summary statistics.')
IOfile.add_argument('--ld_snp_fp', default='geno.bim', type=str, 
    help='File path of the LD SNP information, which is used for alignment of the LD SNPs with the GWAS SNPs. The required columns are CHR, BP and SNP. In our study, we simply use the .bim file for this alignment purpose. Other file formats are also allowed, as long as the required columns are present.')
IOfile.add_argument('--ld_fp', default='ld_mat', type=str, 
    help='File paths of the LD matrix. Should be stored in .npz format. If --partition is specified, the filepath should contain "@" to indicate where --block_index should be inserted.')
IOfile.add_argument('--LD_approx_path', default=None, type=str,
    help='File path to pre-calculated sparse representation or approximation of the LD.')
IOfile.add_argument('--pheno_index', default="1", type=str, 
    help='Index number of the experiment replicate. (Used most for simulation purposes).')
IOfile.add_argument('--N', default=None, type=int, 
    help='Sample size of the GWAS sample. Required for HEELS estimation.')
# IOfile.add_argument('--N_ref', default=None, type=int, 
    # help='Sample size of the reference sample for LD. Required. ')
IOfile.add_argument('--stream-stdout', default=False, action="store_true", help='Stream log information on console in addition to writing to log file.')

# block-wise 
IOfile.add_argument('--partition', default=False, action="store_true", 
    help='Whether to run block-wise HEELS estimation.')
IOfile.add_argument('--num_blocks', default=None, type=int, 
    help='Number of blocks to run partitioned-HEELS, to replace "@" in --ld_fp for reading of LD blocks. Only used when --partition is turned on.')

# plink input (may not be as reliable)
# IOfile.add_argument('--plink_ld_bin_full', default=None, type=str, 
#     help='File path of the r values saved in plink ld.bin format. (Single file with no blocks.)')
# IOfile.add_argument('--plink_ld_band', default=None, type=str, 
    # help='File paths of the banded LD r values in (dense) plink.ld format. Caution that this LD file may not contain all pairwise LD information. NOT RECOMMENDED for use.')
# IOfile.add_argument('--plink_ld_bin_block', default=None, type=str, 
    # help='File paths of the r values, saved in plink ld.bin format. Todo: If the filename prefix contains the symbol @, will replace the @ symbol with all available files in the directory.')

## LD approximation flags
LD_approx = parser.add_argument_group(title="Flags related to LD approximation")

# key parameters 
LD_approx.add_argument('--LD_approx_B', default=300, type=int, 
    help='Central bandwidth of the banded component. For hyperparameter selection, this flag specifies the starting value of the bandwidth.')
LD_approx.add_argument('--LD_approx_R', default=200, type=int, 
    help='Number of low-rank factors for the off-centralband component. For hyperparameter selection, this flag specifies the starting value of the number of low rank factors.')
LD_approx.add_argument('--LD_approx_mode', default=None, type=str,
    help='If None, no LD decomposition will be performed. If "auto", grid search of the (b,r) values; if "manual", single decomposition of the LD using the specified (b,r) values.')
LD_approx.add_argument('--LD_approx_method', default=None, type=str,
    help='The specific method to be used for decomposing the LD. Current options include: joint, seq_band_lr, seq_lr_band, PSD_band_lr, lr_PSD_band, spike_LR, spike_LR_joint, spike_PCA, Band_only, LR_only.')
LD_approx.add_argument('--LR_decomp_method', default="optim", type=str,
    help='The mode to conduct low-rank decomposition. Mode code: optim - use optimization (avoid eigen, recommended); random_svd - use randomized algorithm (avoid eigen); exact - direct eigendecomposition (can be quite slow).')
LD_approx.add_argument('--method', default="L-BFGS-B", type=str, 
    help='Method for joint optimization of low-rank and banded part of LD')


# hyperparameter tuning / optimization related
LD_approx.add_argument('--LD_approx_err_tol', default=0.1, type=float, 
    help='The tolerance for the LD approximation error (in ratio of matrix norm). Used in low-rank component approximation as a stopping criterion.')
LD_approx.add_argument('--LD_approx_ftol', default=1e-3, type=float, 
    help='The stopping criterion argument for the LR approximation optimizer')
LD_approx.add_argument('--LD_approx_inc_SVD', default=False, action="store_true",
 help='Whether or not to use the incremental SVD for seq and PSD_band strategies. If applied, only the optimized R value will be used for CV.')
LD_approx.add_argument('--LD_approx_num_valid_exp', default=50, type=int, 
    help='The number of validation experiments to use for each (B,R) setting.')
LD_approx.add_argument('--LD_approx_valid_bias_tol', default=0.01, type=float, 
    help='The convergence criterion for the h2 estimates from the paired values.')
LD_approx.add_argument('--LD_approx_valid_h2', default=0.5, type=float, 
    help='Value of heritability (Oracle). Used for validation of the LR decomposition method.')
LD_approx.add_argument('--LD_approx_valid_sumstats_fp', default=None, type=str,
    help='File path to the Z-statistics from synthetic phenotypes for validation. If not specified, but hyperparameter tuning is requested, new sumstats will be generated. ')
LD_approx.add_argument('--LD_approx_valid_YtY', default=None, type=str,
    help='File path to the sample variance of the synthetic phenotypes for validation.')
LD_approx.add_argument('--CV_metric', default=None, type=str,
    help='The cross-validation metric to be used for selecting optimal values. Either bias or mse.')
LD_approx.add_argument('--min_sparsity', default=0.07, type=float, 
    help='Upper bound of the sparsity allowed for LD approximation, i.e. B, R < min_sparsity * number of variants')
LD_approx.add_argument('--B_step_size', default=100, type=int,
    help='Step size for increasing the bandwidth of the banded matrix in pseudovalidation.')
LD_approx.add_argument('--R_step_size', default=50, type=int,
    help='Step size for increasing the number of low-rank factors in pseudovalidation.')


# misc. method-specified adjustment
LD_approx.add_argument('--LD_approx_thres_eigvec', default=0, type=float, 
    help='The value below which eigenvectors are assumed as zero. (Used for further compressing of LD representation).')
LD_approx.add_argument('--resid_sigma_estMethod', default=3, type=int,
    help='The mode to use for pre-estimating residual sigma for spiked model. Mode code: 1-top submatrix; 2-random sub-sampling; 3-average of 10 sub-sampling. (Only used for spiked models!)')
LD_approx.add_argument('--spike_LR_diag', default="homo", type=str,
    help='The mode to approximate the diag component of the spike + LR model. Mode code: homo - single parameter, identical diagonal elements; hetero - multiple parameters, different diagonal elements.')

# legacy from old initial experiments
LD_approx.add_argument('--approx_band', default=None, type=int, 
    help='Bandwidth of the banded matrix used to approximate the LD. (Not compatible with low-rank approximation).')
LD_approx.add_argument('--approx_band_trace_only', default=None, type=int,
    help='The bandwidth of the banded matrix used to approximate the LD matrix FOR the TRACE step only.')
LD_approx.add_argument('--precond_band', default=None, type=int, 
    help='The bandwidth of the banded matrix used to approximate Wt, FOR preconditioning only.')

# projection of the Z-stats
LD_approx.add_argument('--LD_approx_Zproj', default=None, type=str,
    help='The mode to use for projecting the marginal statistics onto the eigenspace of the approximated LD matrix. Two options enabled: eigendecomposition or least-square solver.')
LD_approx.add_argument('--LD_approx_tsvd_R', default=None, type=int,
    help='The low-rank number for the full approximating LD.')

# LD_approx.add_argument('--LD_approx_adj_banded_PSD', default=None, type=str,
#     help='Method to adjust for the NPD of the banded part. Two options are enabled - trace_match, boost_eigval. The second option is only usable in conjunction with Zproj (so that the full approx matrix is re-eigendecomped).')
# LD_approx.add_argument('--lowrank_calc_eigs', default=False, action="store_true", 
#     help='Whether or not to (newly) calculate eigenvectors and eigenvalues')
# LD_approx.add_argument('--lowrank_eigs', default=None, type=str, 
#     help='Filepath to save the (full) set of eigenvalues and eigenvectors')

LD_approx.add_argument('--threshold_r', default=None, type=float, 
    help='Assign zero to elements of the R matrix that is below this threhold value.')

## Estimation parameters
estimation = parser.add_argument_group(title="Misc. flags related to estimation")

estimation.add_argument('--not_run_heels', default=False, action="store_true",
 help='Debug flag for isolating the LR-approximation step from the actual estimation step')
estimation.add_argument('--init_values', default=None, type=str, 
    help='Initial values of sigma_g^2 and sigma_e^2. If not specified, these values are randomly generated.')
estimation.add_argument('--constrain_sigma', default=False, action="store_true", 
    help='Whether to rescale sigma_g^2, sigma_e^2 at each run, so that they sum up to 1.'),
estimation.add_argument('--calc_var', default=False, action="store_true", 
    help='Whether to calculate and report standard error of the estimates.')
estimation.add_argument('--use_lrt', default=False, action="store_true", 
    help='Whether to use LRT to conduct inference. p-value is derived from chi-square distribution')
estimation.add_argument('--lrt_mode', default="mixture", type=str, 
    help='The type of null distribution to use for calibrating the LRT statistic. Options: mixture - 0.5,0.5 combination of chi2 with DF 1 and 2.')
estimation.add_argument('--maxIter', default=100, type=int, 
    help='Maximum number of iterations for HEELS.')
estimation.add_argument('--tol', default=1e-4, type=float, 
    help='The tolerance for convergence criterion.')

# related to updating equations
estimation.add_argument('--approx_YtY_bs', default=False, action="store_true", 
    help='Whether or not to use effect size, standard error and alele frequency in the summary statistics to approximate the sample variance of y.')
estimation.add_argument('--beta_name', default="BETA", type=str,
    help='Column name of marginal statistics. Used for approximating the sample variance (y\'y/n). ')
estimation.add_argument('--se_name', default="SE", type=str,
    help='Column name of the standard error of the marginal statistics. Used for approximating the sample variance (y\'y/n)')
estimation.add_argument('--freq_name', default="MAF", type=str,
    help='Column name of the allele frequencies. Used for rescaling the LD matrix when using the original scale.')
estimation.add_argument('--z_name', default='Z', type=str,
    help='Column name of the Z statistics in the sumstats. Used for approximating the sample variance (y\'y/n)')
estimation.add_argument('--n_name', default='n', type=str,
    help='Column name of sample size. Used for approximating the sample variance (y\'y/n)')
estimation.add_argument('--YtY', default=1, type=float, 
    help='Sample variance input (specify if known). ')
estimation.add_argument('--update_sigma_g', default="Seq", type=str, 
    help='Type of updating equation for sigma_g. Default option uses the working value of sigma_g^2 from the previous iteration. Alternative formulation exists. ')

# legacy flags from previous experiments
estimation.add_argument('--mahalanobis_kernel', default=False, action="store_true", 
    help='Whether to use the Mahalanobis kernel for the RE estimates'),
estimation.add_argument('--mah_kernel_method', default="eigen", type=str,
    help='Method to adjust the genotypic matrix using Mahalanobis kernel.'),
estimation.add_argument('--randomNormal_N', default=None, type=int,
    help='Number of random normal vectors to use in estimating the trace. Only used if specified')
estimation.add_argument('--hutchinson_N', default=None, type=int, 
    help='Number of Rademacher samples vectors to use in estimating the trace (Hutchinson estimator). Only used if specified')
estimation.add_argument('--sampleBasis_N', default=None, type=int, 
    help='Number of basis vectors to draw, for approximating the trace, using a sample average of e^tW^-1e.')
estimation.add_argument('--update_interval', default=None, type=int, 
    help='Number of iterations before updating the pre-conditioner. (Not in use right now)')
estimation.add_argument('--eigvals_banded', default=False, action="store_true", 
    help='Calculate the eigenvalues of the banded LD matrix (one time) for the trace inverse calculation.')
estimation.add_argument('--ref_Inverse', default=False, action="store_true", 
    help='One-step refinement to the approximating inverse, i.e. inverse of the preconditioner.')
estimation.add_argument('--chol_inverse', default=False, action="store_true",
    help='Use Cholesky factorization to calculate the inverse of W. Used for trace term only.')
estimation.add_argument('--use_QR', default=False, action="store_true", 
    help='Use QR decomposition to calculate the inverse. Used for trace only.')
estimation.add_argument('--use_lobpcg', default=False, action="store_true", 
    help='Use locally optimal block PCG to calculate eigenvalues.')
# estimation.add_argument('--num_top_eigvals', default=None, type=int, 
#     help='Number of top eigenvalues to use for calculating variance. By default, None means all eigvalues will be used.')



# Debugging parameters
diagnostics = parser.add_argument_group(title="Various flags for debugging purposes. Generally should not be used in real applications.")
diagnostics.add_argument('--check_Y_ll', default=False, action="store_true", help='For reconciling the likelihood function based on summary statistics with the likelihood function based on individual-level data.')
diagnostics.add_argument('--std_geno_fp', default=None, type=str, 
    help='File path to the saved standardized genotype file. For DEBUG purpose. Real application should not require this.')
diagnostics.add_argument('--pheno_mat_fp', default=None, type=str, 
    help='File path to the simulated phenotype matrix file. For DEBUG purpose only. Real application should not require this.')
diagnostics.add_argument('--plink_bfile_fp', default=None, type=str, 
    help='File path prefix of the genotypic files of causal variants. Useful for generation of validation sumstats.')


## Operators
if __name__ == '__main__':
    args = parser.parse_args()

     ## Instantiate log file and masthead
    logging.basicConfig(format='%(asctime)s %(message)s', filename=args.output_fp + '.log', filemode='w', level=logging.INFO, datefmt='%Y/%m/%d %I:%M:%S %p')
    if args.stream_stdout:
        logging.getLogger().addHandler(logging.StreamHandler())

    header_sub = header
    header_sub += "Calling ./run_HEELS.py \\\n"
    defaults = vars(parser.parse_args(''))
    opts = vars(args)
    non_defaults = [x for x in opts.keys() if opts[x] != defaults[x]]
    options = ['--'+x.replace('_','-')+' '+str(opts[x])+' \\' for x in non_defaults]
    header_sub += '\n'.join(options).replace('True','').replace('False','')
    header_sub = header_sub[0:-1] + '\n'

    logging.info(header_sub)
    logging.info("Beginning HEELS analysis...")
    start_time = time.time()

    # read in sample sizes
    n_ref = n = int(args.N)
    tol = float(args.tol)

    try:
        # =================
        # parsing LD input 
        # =================
        if args.LD_approx_path is None:
            logging.info("Parsing the full LD matrix from file")

            if args.partition:
                logging.info("Reading in multiple LD blocks.")

                # replace @ with block indexes
                block_index = list(np.arange(int(args.num_blocks)) + 1)
                LD_fp_list = [args.ld_fp.replace("@", str(block_index[i])) for i in range(args.num_blocks)]

                # read in LD as a list of objects
                LD_list = [np.load(x)['LD'] for x in LD_fp_list]
                LD_list = [(LD + np.transpose(LD))/2 for LD in LD_list]

                # count the number of SNPs
                m_ref_list = [LD.shape[0] for LD in LD_list]
                m_total = np.sum(m_ref_list)
                logging.info("Total number of markers: {}".format(m_total))

                # scale LD blocks by N_ref and M
                # LD_list = [LD_list[i] * n_ref / m_ref_list[i] for i in range(len(LD_list))]
                LD_list = [LD_list[i] * n_ref / m_total for i in range(len(LD_list))]

                # drop NaN values (should not do anything in simulations)
                LD_noNA_list = list()
                for i in range(len(LD_list)):
                    LD = LD_list[i]
                    idx = np.where(~np.isnan(LD).any(axis=0))[0]
                    logging.info("Dropping {} SNP due to invalid LD information".format(LD.shape[0] - idx.shape[0]))
                    LD_noNA_list.append(LD[idx[:, np.newaxis], idx])
                LD_list = LD_noNA_list

                K = len(LD_list)

                # put into a list (for HEELS estimation)
                LD_dict_list = [dict({'LD_banded': heels_format.band_format(LD_list[k], LD_list[k].shape[0])}) for k in range(K)]

            else:
                # logging.info("Single LD block (no partitioning)")
                obj= np.load(args.ld_fp)
                LD = obj['LD']
                m_ref = LD.shape[0]
                LD = (LD + np.transpose(LD)) / 2

                # scale LD
                LD = LD * n_ref / m_ref

                # drop NaN values
                idx = np.where(~np.isnan(LD).any(axis=0))[0]
                logging.info("Dropped {} SNP due to invalid LD information".format(LD.shape[0] - idx.shape[0]))
                LD = LD[idx[:, np.newaxis], idx]

                # band the LD matrix (to be consistent with the sparse representation)
                LD_dict = dict({'LD_banded': heels_format.band_format(LD, LD.shape[0])})

            logging.info("LD parsing is complete.")
            logging.info("Number of markers: {}.".format(str(LD.shape[0])))
            logging.info(short_borderline)

            # Decompose the LD matrix
            if args.LD_approx_mode is None:
                logging.info("No low-dimensional LD decomposition is conducted.")
            else:
                if args.partition:
                    raise IOError("Currently, we do not support decomposing multiple LD blocks simultaneously. Please specify the LD blocks separately to find the sparse representation for each of the blocks.")

                else:
                    logging.info("Finding the sparse representation of the LD matrix")
                    logging.info("Approximation mode: {}.".format(args.LD_approx_mode))
                    logging.info("Approximation method: {}.".format(args.LD_approx_method))
                    logging.info(short_borderline)

                    LD_approx_start = time.time()
                    LD_norm = np.linalg.norm(LD)

                    # TO DO: enable flexible specification of the norm function
                    if args.LD_approx_mode == "auto":
                        LD_dict, F_norm = heels_LDdecomp.LDdecomp_auto(args, LD)

                    elif args.LD_approx_mode == "manual":
                        LD_dict, F_norm = heels_LDdecomp.LDdecomp(args, LD)

                    LD_approx_time = round(time.time() - LD_approx_start, 2)
                    logging.info('Sparse approximation time: {T}'.format(T=(LD_approx_time)))

                    if F_norm / LD_norm > 0.1:
                        logging.info("Warning: the percent of LD matrix approximated is below 0.90. This may lead to bias in h2 estimation.")

                    # report the approximation performance of the LD
                    logging.info("Approximated \% of LD: {}%".format(100 - F_norm / LD_norm * 100))

        else:
            logging.info("Reading pre-computed low-dimensionaldimenisn representation of LD from path: \n{}".format(args.LD_approx_path))

            if args.partition: # TO TEST
                block_path_list = args.LD_approx_path.split(',')
                K = len(block_path_list)
                kiu = [read_sparse_LD(args, block_path_list[k], k+1) for k in range(K)]
                LD_dict_list = [x[0] for x in kiu]
                m_ref_list = [x[1] for x in kiu]
            else:
                LD_dict, m_ref = read_sparse_LD(args, args.LD_approx_path)

        logging.info(short_borderline)

        # ======================
        # align X'Y with X'X
        # ======================
        # if HEELS estimates or variance need to be estimated
        if not args.not_run_heels or args.calc_var:
            # read in sumstats (if @ is not specified, read df_ss in whole; if specified, read )

            if "@" not in args.sumstats_fp:
                logging.info("Parsing summary statistics from file")
                df_ss = pd.read_csv(args.sumstats_fp, delim_whitespace=True, index_col=None)

                # check that the sumstats indeed has the Z and n columns
                # (TO DO: acccommodate diff colnames, same as mtag and mama)
                assert 'Z' in list(df_ss.columns), "The summstats is missing a Z column!"
                assert 'n' in list(df_ss.columns), "The sumstats is missing the n column!"

                m0 = df_ss.shape[0]
                logging.info("Number of SNPs in sumstats: {m0}".format(m0 = m0))

                # use the overall average N from sumstats
                n = np.mean(df_ss['n'].values)

            else:
                logging.info("Parsing summary statistics from multiple files")
                block_index = list(np.arange(int(args.num_blocks)) + 1)
                ss_fp_list = [args.sumstats_fp.replace("@", str(block_index[i])) for i in range(args.num_blocks)]
                df_ss, n_avg_block = zip(*[read_sumstats(ss_fp_list[i]) for i in range(args.num_blocks)])
                n = np.mean(n_avg_block)
                df_ss = list(df_ss)

                if not args.partition:
                    df_ss = pd.concat(df_ss)
                    logging.info("Summary statistics are input block-wise.")
                    logging.info("Concatenated into one dataset with dim: {}".format(df_ss.shape))

            # take the overlap between LD and sumstats
            if args.partition: # TO TEST
                ld_snp_fp_list = [args.ld_snp_fp.replace("@", str(block_index[i])) for i in range(args.num_blocks)]
                K = args.num_blocks
                kiu = [align_sumstats_LD(args, ld_snp_fp_list[k], df_ss[k], LD_dict_list[k], m_ref_list[k], k+1) for k in range(K)]
                LD_dict_list = [x[0] for x in kiu]
                df_both_list = [x[1] for x in kiu]

            else:
                LD_dict, df_both = align_sumstats_LD(args, args.ld_snp_fp, df_ss, LD_dict, m_ref)

            logging.info(short_borderline)
        # ==========================
        # prepare for running HEELS
        # ==========================
        if not args.not_run_heels or args.calc_var:
            logging.info("Preparing the Z-scores for running HEELS and adjust if necessary")

            # use beta and se to approx var(y)
            if args.approx_YtY_bs:
                args.beta = np.asarray(df_both[[args.beta_name]])
                args.se = np.asarray(df_both[[args.se_name]])
                args.freq = np.asarray(df_both[[args.freq_name]])
                args.n = np.asarray(df_both[[args.n_name]])
                # args.beta = df_both[[args.z_name]].values / np.sqrt(df_both[[args.n_name]].values)
                # args.se = 1 / np.sqrt(df_both[[args.n_name]].values)
                # args.beta = df_both[[args.z_name]].values / math.sqrt(n)
                # args.se = 1 / math.sqrt(n)

            # standardize sumstats
            if args.partition: # TO TEST
                m_list = [df_both_list[k].shape[0] for k in range(K)]
                m_total = np.sum(m_list)
                # allow markers to have different sample size
                Z_m_list = [np.multiply(df_both_list[k][['Z']].values, np.sqrt(df_both_list[k][['n']].values)) / math.sqrt(m_total) for k in range(K)]
            else:
                m = df_both.shape[0]
                # allow markers to have different sample ize
                Z_m = np.multiply(df_both[[args.z_name]].values, np.sqrt(df_both[[args.n_name]])) / math.sqrt(m) 

                # if args.approx_YtY_bs or args.YtY is not None:
                #     logging.info("Using the original marginal statistics")
                #     Z_m = df_both[[args.beta_name]].values * n_ref * math.sqrt(1 / m)
                #     MAF_vec = df_both[[args.freq_name]].values
                #     X_SD_vec = 1 / np.sqrt(2*np.multiply(MAF_vec, 1-MAF_vec))

                #     logging.info("Adjusting LD matrix, using var(X)")
                #     LD_adj_mat = np.outer(X_SD_vec, X_SD_vec)
                #     LD_adj_mat = heels_format.band_format(LD_adj_mat, LD_adj_mat.shape[0])
                #     LD_dict['LD_banded'] = np.multiply(LD_adj_mat, LD_dict['LD_banded'])

                #     # DEBUG: identify NaN and Inf values
                #     ar_nan = np.where(np.isnan(LD_dict['LD_banded']))
                #     ar_inf = np.where(np.isinf(LD_dict['LD_banded']))
                #     logging.info("NaN values: ")
                #     logging.info(ar_nan[0].shape)
                #     logging.info("Inf values: ")
                #     logging.info(ar_inf[0].shape)
                # else:
                # Z_m = df_both[[args.z_name]].values * math.sqrt(n_ref / m)        

            # initialize var component values
            if args.init_values is not None:
                sigma_g_0, sigma_e_0 = [float(x) for x in args.init_values.split(',')]
            else:
                # random.seed(args.pheno_index)
                sigma_g_0 = np.random.uniform(size=1)[0]
                sigma_e_0 = 1 - sigma_g_0

            # ====================================================================
            # Projecting X'Y
            # (not used for standard estimation, but may be useful for diagnostics)
            # e.g. checking the small or negative eigenvalues
            # currently implementation only allows for single LD block
            # ====================================================================
            if args.LD_approx_Zproj is not None:
                # DEBUG
                logging.info("Z scores before projecting: ")
                logging.info(Z_m)

                # if TSVD R value is specified, second-layer of approximation
                LD_approx_banded = heels_format.band_format(LD_approx, LD_approx.shape[0])
                if args.LD_approx_tsvd_R is not None:
                    logging.info("Projecting the Z-statistics onto the eigenspace of the LD approximation")
                    tsvd_R = int(args.LD_approx_tsvd_R)
                    
                    logging.info("Conducting TSVD with {} factors".format(tsvd_R))
                    w_approx, v_approx = eig_banded(LD_approx_banded, lower = True, select = 'i', select_range = (m-tsvd_R, m-1))
                else:
                    w_approx, v_approx = eig_banded(LD_approx_banded, lower = True)

                eigval = w_approx[::-1]
                eigvec = v_approx[:, ::-1]

                # if args.LD_approx_adj_banded_PSD == "boost_eigval": # not used
                #     logging.info("Boosting the eigenvalue of the full approximating matrix")
                #     eigval = eigval + np.abs(np.min(eigval))

                # the following can also be used to output diagnostics (eigvals) of LD approx
                logging.info("Smallest eigenvalues of FULL approx LD: {}".format(eigval[-10:]))
                logging.info("Number of negative eigval: {}".format(np.sum(eigval < 0)))

                # project Z_m onto the eigen space
                Z_m = eigvec.dot(np.transpose(eigvec)).dot(Z_m)
            
                # DEBUG
                logging.info("Z scores after projecting: ")
                logging.info(Z_m)

                if args.LD_approx_path is None: # full LD information is provided
                    LD_lr = eigvec.dot(np.diag(eigval).dot(np.transpose(eigvec)))
                    F_norm = np.linalg.norm(LD - LD_lr)

                    # compute F-norm and report approximating performance 
                    logging.info("After the 2nd layer of approximation.")
                    logging.info("Approximated \% of LD: {}%".format(100 - F_norm / LD_norm * 100))
                

        logging.info(short_borderline)
        # =============
        # Running HEELS
        # =============
        if not args.not_run_heels:

            if args.partition: # TO TEST
                logging.info("Start running partitioned-HEELS")
                sigma_g, sigma_e, i, sigma_g_list, sigma_e_list = run_HEELS_block(args, Z_m_list, LD_dict_list, sigma_g_0, sigma_e_0, m_list, n, YtY=args.YtY, update_sigma_g="Seq", tol=tol, maxIter=1000)
           
                # logging.info("DEBUG: Start concatenating block-wise LD together")
                # LD = np.zeros(shape=(m_total, m_total))
                # Z_index = np.insert(np.cumsum(np.asarray(m_list)), 0, 0) 

                # for k in range(K):
                #     LD[Z_index[k]:Z_index[k+1], Z_index[k]:Z_index[k+1]] = LD_list[k]

                # # band the LD matrix (to be consistent with the sparse representation)
                # LD_dict = dict({'LD_banded': heels_format.band_format(LD, LD.shape[0])})

                # Z_m = np.vstack(Z_m_list)
                # logging.info("Shape of concatenated Z df: {}".format(Z_m.shape))

                # sigma_g, sigma_e, i, sigma_g_list, sigma_e_list, ll_path, Y_ll_path = run_HEELS(args, Z_m, LD_dict, sigma_g_0, sigma_e_0, m_total, n, YtY=args.YtY, update_sigma_g="Seq", tol=tol, maxIter=1000)

            else:
                logging.info("Start running HEELS")
                sigma_g, sigma_e, i, sigma_g_list, sigma_e_list = run_HEELS(args, Z_m, LD_dict, sigma_g_0, sigma_e_0, m, n, YtY=args.YtY, update_sigma_g="Seq", tol=tol, maxIter=100)

            logging.info("Finish running HEELS. Writing results into file.")
            logging.info(short_borderline)

            # save to files
            with open(args.output_fp + ".txt", 'w') as file:
                file.write(str(args.pheno_index) + '\t' + "sigma_g^2" + '\t' + str(sigma_g) + "\n" + str(args.pheno_index) + '\t' + "sigma_e^2" + '\t' + str(sigma_e) + "\n")

            # if not args.partition:
            #     with open(args.output_fp + "_ll_path.txt", 'w') as file:
            #         for ll in ll_path:
            #             file.write(str(ll)+'\n')

            #     if args.check_Y_ll:
            #         with open(args.output_fp + "_Y_ll_path.txt", 'w') as file:
            #             for ll in Y_ll_path:
            #                 file.write(str(ll)+'\n')

        elif args.calc_var:
            logging.info("--not_run_heels is specified, so no HEELS performed.")
            logging.info("Loading pre-computed HEELS estimates for variance computation.")
            
            if args.est_fp is None:
                args.est_fp = args.output_fp

            est = pd.read_csv(args.est_fp + ".txt" , delim_whitespace=True, index_col=None, names=['pheno', 'component', 'estimates'])

            sigma_g = est.loc[0, 'estimates']
            sigma_e = est.loc[1, 'estimates']

        # =============
        # Variance
        # =============
        
        if args.calc_var:
            
            logging.info(short_borderline)
            
            if args.partition:
                raise IOError("Currently, variance can only be calculated for a single block.")

            # Inference based on likelihood ratio test
            if args.use_lrt:
                from scipy.stats import chi2
                # compute likelihood under the null
                null_logL = heels_utils.compute_logL(1, m, n, Z_m, LD_dict['LD_banded'], null=True)
                logging.info("LogL under H0: {}".format(null_logL))

                # compute likelihood under the alternative (estimated value)
                lam = sigma_e / sigma_g
                alt_logL = heels_utils.compute_logL(lam, m, n, Z_m, LD_dict['LD_banded'], sigma_e, 1)
                logging.info("LogL under H1: {}".format(alt_logL))

                # conduct LRT:
                chi_stat = 2*(alt_logL - null_logL)

                # inference based on mixture of distribution
                if args.lrt_mode == "mixture":
                    logging.info("Conducting LRT using a mixture of two chi-square distributions")
                    logging.info("LRT statistics: {}".format(chi_stat))
                    chi_dummy = np.hstack((chi2.rvs(df=1, size=int(5e7)), chi2.rvs(df=2, size=int(5e7))))
                    pval = np.mean(chi_dummy > chi_stat)
                else: # assume no boundary problems - no mixture
                    logging.info("Conducting LRT using statistic {} and DF {}".format(chi_stat, m+2))
                    pval = 1 - chi2.cdf(chi_stat, df = m+2)

                logging.info("P-value is: {0:.5f}".format(pval))


            # Inference using normal approximation of the Wald statistic
            else:
                if args.LD_approx_method is None:
                    logging.info("Computing the variance of HEELS estimator")
                    # LD_banded = heels_format.band_format(LD, LD.shape[0])
                    # eigs = eigvals_banded(LD_banded, lower = True)
                    eigs = np.linalg.eigvalsh(LD)
                    m = LD.shape[0]

                    # # DEBUG
                    # logging.info("Eigenvalues of the LD (approximation of LD): ")
                    # logging.info(eigs)
                    # logging.info("Number of eigenvales: {}".format(eigs.shape[0]))

                    logging.info("Calculating the SE using the analytical form (ll based)")
                    var_mat_heels = heels_utils.HEELS_variance(sigma_g, sigma_e, eigs, n, m)

                else:
                    if args.LD_approx_method == "Band_only" or args.LD_approx_method == "LR_only":
                        raise ValueError("Currently, with sparse representation of the LD, only the Banded + LR form is enabled for computing the variance")

                    logging.info("Computing the variance of HEELS estimator using approx of LD")
                    var_mat_heels = heels_utils.HEELS_variance_lowrank(sigma_g, sigma_e, LD_dict['LD_banded'], LD_dict['eigval'], LD_dict['eigvec'], n, m)

                # multi Delta function for var of h2
                grad_denom = (sigma_g + sigma_e)**2
                grad_vec = np.asarray([(-1)* sigma_g / grad_denom, sigma_e / grad_denom])
                heels_se_est = np.sqrt(np.sum(np.multiply(np.einsum('i,ij->j', grad_vec, var_mat_heels), grad_vec)))
                heels_se_g = np.sqrt(var_mat_heels[1,1])
                heels_se_e = np.sqrt(var_mat_heels[0,0])
                logging.info("HEELS SE of h2: {}".format(heels_se_est))
                logging.info("HEELS Asymptotic SE of sigma_g: {}".format(heels_se_g))
                logging.info("HEELS Asymptotic SE of sigma_e: {}".format(heels_se_e))
                logging.info("HEELS Asymptotic SE of h^2: {}".format(heels_se_est))

                # DEBUG (plugging in true values)
                # var_mat_true = heels_utils.HEELS_variance(0.5, 0.5, eigs, n, m)
                # heels_se_g_true = np.sqrt(var_mat_true[1, 1])
                # heels_se_e_true = np.sqrt(var_mat_true[0, 0])
                # logging.info("HEELS true SE of sigma_g: {}".format(heels_se_g_true))
                # logging.info("HEELS true SE of sigma_e: {}".format(heels_se_e_true))

                # save to files
                with open(args.output_fp + "_se.txt", 'w') as file:
                    file.write(str(args.pheno_index) + '\t' + "sigma_g^2" + '\t' + str(heels_se_g) + "\n" + str(args.pheno_index) + '\t' + "sigma_e^2" + '\t' + str(heels_se_e) + "\n" + str(args.pheno_index) + '\t' + "h^2" + '\t' + str(heels_se_est) + "\n")

    except Exception as e:
        logging.error(e,exc_info=True)
        logging.info('Analysis terminated from error at {T}'.format(T=time.ctime()))
        time_elapsed = round(time.time() - start_time, 2)
        logging.info('Total time elapsed: {T}'.format(T=heels_utils.sec_to_str(time_elapsed)))

    logging.info('Total time elapsed: {}'.format(heels_utils.sec_to_str(time.time()-start_time)))
    logging.info(borderline)
