#!/usr/bin/env python

#-------------------------------------------------------
# Summary Statistics and Linkage Disequilibrium

# Find the sparse representation of the LD matrix 

# Required modules:
# - utils/heels_format

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
# sys.path.append('/n/holystore01/LABS/xlin/Lab/huili/HEELS/main_func/utils')
# sys.path.append('/n/holystore01/LABS/xlin/Lab/huili/HEELS/main_func')
from utils import heels_utils
from utils import heels_format
import run_HEELS

borderline = "<><><<>><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>"
short_borderline = "<><><<>><><><><><><><><><><><><><><><><><><><><><><><><><><><><>\n"


# Banding and LR functions
def LR_func(LD_0, LD_B, r):
    # LD_B is in dense format
    delta_LD = LD_0 - LD_B
    LD_delta_banded = heels_format.band_format(delta_LD, delta_LD.shape[0])
    m = delta_LD.shape[0]
    # select the top r eigenvectors + eigenvalues
    w, v = eig_banded(LD_delta_banded, lower = True, select = 'i', select_range = (m-r, m-1))
    return w, v

def Band_func(args, LD_0, w, v):
    LD_lr = eig_construct(w, v, args.LD_approx_R)
    delta_LD = LD_0 - LD_lr
    LD_B = heels_format.band_format(delta_LD, args.LD_approx_B)
    # LD_B is in banded-matrix format
    return LD_B

def eig_construct(w, v, R):
    eigval = w[-R:][::-1]
    eigvec = v[:, -R:][:, ::-1]
    LD_lr = eigvec.dot(np.diag(eigval).dot(np.transpose(eigvec)))
    return LD_lr

def LR_func_optim(LD_0, LD_b, r, method='optim', min_method='L-BFGS-B', LD_approx_ftol=1e-3):

    '''
    Low-rank decomposition of the residual off-centralband matrix.

    Options include (specified by argument 'method'):
    -- optim: use optimization to solve for the LR component (approximate)
    -- random_svd: use randomized SVD to solve for the LR component (approximate)
    -- exact: direct eigendecomposition (exact)

    '''
    logging.info("Starting low-rank decomposition...")
    delta_LD = LD_0 - LD_b
    target_norm  = np.linalg.norm(delta_LD)
    m = LD_0.shape[0]

    if method == "optim":
        U0 = np.random.rand(m, r)
        logging.info("Using optimization")
        opt = minimize(approx_err_LR, x0 = np.asarray(list(U0.flatten())), args = (delta_LD, r, target_norm), jac = grd_U_LR, method = min_method, options = {'ftol': float(LD_approx_ftol)})

        # construct solution from the optimization output
        U = rebuild_U_mat(opt.x, m, r)
        LD_lr = U.dot(np.transpose(U))
        u, s, vh = np.linalg.svd(U, full_matrices = False, compute_uv = True)
        eigval = (s**2)[:r]
        eigvec = u[:, :r]

    elif method == "exact":
        LD_delta_banded = heels_format.band_format(delta_LD, delta_LD.shape[0])
        logging.info("Using exact eigen-decomposition")

        # select the top r eigenvectors + eigenvalues
        w, v = eig_banded(LD_delta_banded, lower = True, select = 'i', select_range = (m-r, m-1))
        eigval = w[-r:][::-1]
        eigvec = v[:, -r:][:, ::-1]

    elif method == "random_svd":
        logging.info("Using randomized SVD")
        eigs = skmath.randomized_svd(delta_LD, n_components=r, n_oversamples=2*m-r) #min([2*m-r, 2000])
        eigvec = eigs[0]
        eigval = eigs[1]

    # logging.info("Finish low-rank decomposition")

    return eigval, eigvec

# Joint optimization
def approx_err(params, LD, R, B, LD_norm):

    '''Objective function for solving both the banded and low-rank components'''

    M = LD.shape[0]
    U = rebuild_U_mat(params[:(M*R)], M, R)
    L = rebuild_chol_mat(params[(M*R):], M, B)
    F_norm = np.linalg.norm(LD - U.dot(np.transpose(U))- L.dot(np.transpose(L)))
    # logging.info(F_norm)
    # logging.info(F_norm / LD_norm)

    return F_norm**2

def grd_U(params, LD, R, B, LD_norm):

    '''Gradient of the objective function for solving the banded + low-rank components'''

    M = LD.shape[0]
    U = rebuild_U_mat(params[:(M*R)], M, R)
    L = rebuild_chol_mat(params[(M*R):], M, B)
    dfdU = -2*(LD - L.dot(np.transpose(L)) - U.dot(np.transpose(U))).dot(U)
    dfdL = -2*(LD - L.dot(np.transpose(L)) - U.dot(np.transpose(U))).dot(L)

    return np.asarray(tuple(dfdU.flatten()) + flatten_chol(dfdL, B))

# Banded optimization
def approx_err_banded(params, LD_b, B, LD_b_norm):

    '''Objective function for solving the banded component'''

    M = LD_b.shape[0]
    L = rebuild_chol_mat(params, M, B)
    F_norm = np.linalg.norm(LD_b - L.dot(np.transpose(L)))
    # logging.info(F_norm)
    # logging.info(F_norm / LD_b_norm)

    return F_norm**2 

def grd_U_banded(params, LD_b, B, LD_b_norm):

    '''Gradient of the objective function for solving the banded component'''

    M = LD_b.shape[0]
    L = rebuild_chol_mat(params, M, B)
    dfdL = -2*(LD_b - L.dot(np.transpose(L))).dot(L)

    return np.asarray(flatten_chol(dfdL, B))

# LR optimization
def approx_err_LR(params, LD, R, LD_norm):

    '''Objective function for solving the low-rank component'''

    M = LD.shape[0]
    U = rebuild_U_mat(params, M, R)
    F_norm = np.linalg.norm(LD - U.dot(np.transpose(U)))
    # logging.info(F_norm)
    # logging.info(F_norm / LD_norm)

    return F_norm**2

def grd_U_LR(params, LD, R, LD_norm):

    '''Gradient of the objective function for solving the low-rank component'''

    M = LD.shape[0]
    U = rebuild_U_mat(params[:(M*R)], M, R)
    dfdU = -2*(LD - U.dot(np.transpose(U))).dot(U)
    gradvec = np.asarray(list(dfdU.flatten()))

    return gradvec

# Spiked LR optimization 
def approx_err_spike_LR(params, LD, R, LD_norm, spike_LR_diag):

    '''Objective function for solving the low-rank component in the spiked covariance model'''

    M = LD.shape[0]
    U = rebuild_U_mat(params[:(M*R)], M, R)

    if spike_LR_diag == "homo":    
        resid_sigma = params[M*R]
        F_norm = np.linalg.norm(LD - U.dot(np.transpose(U))- np.eye(M)*resid_sigma)
    
    elif spike_LR_diag == "hetero":
        resid_sigma = params[M*R:]
        F_norm = np.linalg.norm(LD - U.dot(np.transpose(U))- np.diag(resid_sigma))
    
    # logging.info(F_norm)
    # logging.info(F_norm / LD_norm)

    return F_norm**2

def grd_U_spike_LR(params, LD, R, LD_norm, spike_LR_diag):
    M = LD.shape[0]
    U = rebuild_U_mat(params[:(M*R)], M, R)

    if spike_LR_diag == "homo":
        resid_sigma = params[M*R]
        dfdU = -2*(LD - np.eye(M)*resid_sigma - U.dot(np.transpose(U))).dot(U)
        dfdsigma = -2*np.sum(np.diag(LD - np.eye(M)*resid_sigma - U.dot(np.transpose(U))))
        gradvec = np.asarray(list(dfdU.flatten()) + [dfdsigma])

    elif spike_LR_diag == "hetero": 
        # NEED TO CHECK IF PARTIAL SIGMA IS CORRECT
        resid_sigma = params[M*R:]
        dfdU = -2*(LD - np.diag(resid_sigma) - U.dot(np.transpose(U))).dot(U)
        dfdsigma = -2*np.diag(LD - np.diag(resid_sigma) - U.dot(np.transpose(U))) #check
        gradvec = np.asarray(list(dfdU.flatten()) + list(dfdsigma)) #check 

    return gradvec

# Util functions for optimization
def flatten_chol(chol_mat, B):
    '''
    Flatten the Chol factor of banded matrix (p x B) into 1d vector for optim

    B: bandwidth of the matrix (and its chol factor)
    '''
    M = chol_mat.shape[0]
    centralBand_ind = np.where(np.tri(M,M, dtype=bool)&~np.tri(M,M,-B,dtype=bool))
    chol_fac_vec = tuple(chol_mat[centralBand_ind])

    # for i in range(M):
    #     for j in range(i): # standardize lower trangular elements
    #         chol_mat[i,j] = chol_mat[i,j]/np.sqrt(chol_mat[i,i]*chol_mat[j,j])
    # chol_mat[np.diag_indices(M)] = np.log(np.diag(chol_mat))
    # lowTr_ind = np.asarray(np.tril_indices(M))
    
    return chol_fac_vec

def rebuild_U_mat(U_elems, p, R):
    '''
    Construct the Chol factor of the low-rank (pxr) matrix, from 1d vector

    Row-wise filling. Returns a p x R matrix

    '''
    U_mat = np.asarray(U_elems).reshape(p, R)

    return U_mat

def rebuild_chol_mat(chol_elems, p, B):
    '''
    Construct the Chol factor of the banded (pxp) matrix, from 1d vector

    Row-wise filling. Returns a pxp lower-triangular (incl. diag.) matrix

    '''
    cholL = np.zeros((p,p))
    cholL[np.where(np.tri(p,p, dtype=bool)&~np.tri(p,p,-B,dtype=bool))] = np.array(chol_elems)

    # cholL[np.diag_indices(M)] = np.exp(np.diag(cholL))
    # for i in range(M):
    #     for j in range(i): # multiply by exponentiated diags
    #         cholL[i,j] = cholL[i,j]*np.sqrt(cholL[i,i]*cholL[j,j])
    return cholL

def PSD_adjust(B, LD_banded, diag_inflator = 1):

    '''Inflate diagonal to turn a matrix into PSD'''

    m = LD_banded.shape[1]
    while True:
        logging.info("Current value of the diagonal inflator: {}".format(diag_inflator))
        diag_inflator = diag_inflator * 1.5
        if diag_inflator > 1e3:
            logging.info("Failed to adjust for NPD using diagonal inflator (<100), switch to random initialization")
            L0 = flatten_chol(np.random.rand(m, m), B)
            break

        else:
            try:
                LD_banded[0,:] = LD_banded[0,:] + diag_inflator 
                chol_banded= cholesky_banded(LD_banded, lower = True)
                chol0 = heels_format.unband_lower_tri(chol_banded)
                L0 = flatten_chol(chol0, B)
                break
            except:
                pass
    return L0

# Fast PCA related methods
def fast_pca_helper(u_list, p_final, q, r):
    m = u_list[0].shape[0]
    L = len(u_list)
    omega = np.random.normal(size = (m, p_final))

    for qq in range(q):
        Y = 0 
        for ll in range(L):
            Y_l = np.transpose(u_list[ll]).dot(omega)
            Y_l = u_list[ll].dot(Y_l)
            Y = Y + Y_l
        Y = Y / L
        omega = Y

    u, s, vh = np.linalg.svd(Y, full_matrices = False, compute_uv = True)

    return(u[:,:r])

def fast_pca(LD, r, p_sub, p_final, L=100, q=50):
    m = LD.shape[0]
    u_list = list()
    for l in range(L):
        G_l = np.random.normal(size = (m, p_sub))
        Y_l = LD.dot(G_l)
        u, s, vh = np.linalg.svd(Y_l, full_matrices = False, compute_uv = True)
        u_list.append(u)

    eigvec = fast_pca_helper(u_list, p_final, q, r) # mxr matrix
    eigval = np.einsum('jr,jr->r', eigvec, np.einsum('mr,mj->jr', eigvec, LD)) / np.einsum('mk,mk->k', eigvec, eigvec)

    return eigvec, eigval

def est_resid_sigma_subsample(LD, rr, m, seed=7621):
    np.random.seed(seed)
    r_index = np.random.randint(low = 0, high = m, size=rr, dtype=int)
    LD_pre = LD[r_index[:, np.newaxis], r_index]
    w_pre, v_pre = eig_banded(heels_format.band_format(LD_pre, rr), lower = True)
    eigval = np.sort(w_pre)[::-1]

    # capping the diagonal element to be larger than 1
    resid_sigma = np.nanmax([1, np.nanmin(eigval)])
    # resid_sigma = np.nanmin(eigval)
    return resid_sigma

# LD approx error calibration
def joint_opt_approx_err(f_args):
    args, LD, B, R, ftol = f_args
    LD_norm = np.linalg.norm(LD)
    m = LD.shape[0]

    # initialize low-rank
    np.random.seed(7621)
    U0 = np.random.rand(m, R)
    L0 = flatten_chol(np.random.rand(m, m), B)

    # LD_banded = heels_format.band_format(LD, B)
    # L0 = PSD_adjust(B, LD_banded)   

    opt = minimize(approx_err, x0 = np.asarray(tuple(U0.flatten()) + L0), args = (LD, R, B, LD_norm), jac = grd_U, method = args.method, options = {'ftol': ftol})

    # read optimal results
    U = rebuild_U_mat(opt.x[:(m*R)], m, R)
    L = rebuild_chol_mat(opt.x[(m*R):], m, B)
    LD_b = L.dot(np.transpose(L))
    LD_lr = U.dot(np.transpose(U))

    # use SVD on U to obtain optimal eigs
    u, s, vh = np.linalg.svd(U, full_matrices = False, compute_uv = True)
    eigval = (s**2)[:R]
    eigvec = u[:, :R]
    F_norm = np.linalg.norm(LD - LD_lr - LD_b)

    error_prop = F_norm / LD_norm

    logging.info("Prop of approx err with (B,R) = ({}, {}): {}".format(B,R,error_prop))

    return F_norm, error_prop, LD_b, eigval, eigvec

def Band_LR_opt_approx_err(f_args):
    args, LD, B, R, ftol = f_args
    LD_norm = np.linalg.norm(LD)
    m = LD.shape[0]

    # banding the LD matrix
    LD_b = heels_format.dense_band(LD, B)

    # decompose the residual
    w, v = LR_func(LD, LD_b, R)
    eigval = w[-R:][::-1]
    eigvec = v[:, -R:][:, ::-1]

    # calculate approx error
    LD_lr = eigvec.dot(np.diag(eigval).dot(np.transpose(eigvec)))
    F_norm = np.linalg.norm(LD - LD_lr - LD_b)

    error_prop = F_norm / LD_norm

    logging.info("Prop of approx err with (B,R) = ({}, {}): {}".format(B,R,error_prop))

    return F_norm, error_prop, LD_b, eigval, eigvec

def Band_LR_psd_opt_approx_err(f_args):
    args, LD, B, R, ftol = f_args
    LD_b = heels_format.dense_band(LD, B)
    LD_b_norm = np.linalg.norm(LD_b)
    m = LD.shape[0]
    LD_banded = heels_format.band_format(LD_b, B)
    L0 = flatten_chol(np.random.rand(m, m), B)
    # LD_banded = heels_format.band_format(LD_b, B)
    # L0 = PSD_adjust(B, LD_banded)

    # optimize chol factor for the banded part
    opt = minimize(approx_err_banded, x0 = np.asarray(L0), args = (LD_b, B, LD_b_norm), jac = grd_U_banded, method = args.method, options = {'ftol': ftol})

    # reconstruct optimized psd banded matrix
    L = rebuild_chol_mat(opt.x, m, B)
    LD_b_psd = L.dot(np.transpose(L))

    # output errors
    F_norm = np.linalg.norm(LD_b - LD_b_psd)
    error_prop = F_norm / LD_b_norm

    # low-rank decompose the residual
    w, v = LR_func(LD, LD_b, R)
    eigval = w[-R:][::-1]
    eigvec = v[:, -R:][:, ::-1]
    LD_lr = eigvec.dot(np.diag(eigval).dot(np.transpose(eigvec)))

    # calculate the full approx error
    F_norm = np.linalg.norm(LD - LD_lr - LD_b_psd)

    return F_norm, error_prop, LD_b, eigval, eigvec

# Cross-validation functions
def CV_h2_joint(f_args):

    '''
    Running cross-validation to evaluate the performance of the "joint" Banded + LR strategy. 

    CV statistics are calculated for the specified (B,R) hyperparameter setting.

    '''

    args, B, R, LD, LD_norm, m, n, true_h2, df_valid_ss = f_args

    # use (B, R) and get U0 and L0 for init
    np.random.seed(7621)
    U0 = np.random.rand(m, R)
    LD_b = heels_format.dense_band(LD, B)
    LD_banded = heels_format.band_format(LD_b, B)
    L0 = flatten_chol(np.random.rand(m, m), B)
    # LD_banded = heels_format.band_format(LD_b, B)
    # L0 = PSD_adjust(B, LD_banded)

    # optimization joint for both components
    logging.info("Start optimization with (B,R) = ({},{})".format(B, R))
    opt = minimize(approx_err, x0 = np.asarray(tuple(U0.flatten()) + L0), args = (LD, R, B, LD_norm), jac = grd_U, method = args.method, options = {'ftol': float(args.LD_approx_ftol)})

    # read optimal results
    U = rebuild_U_mat(opt.x[:(m*R)], m, R)
    L = rebuild_chol_mat(opt.x[(m*R):], m, B)
    LD_b = L.dot(np.transpose(L))
    LD_lr = U.dot(np.transpose(U))

    # report intermediate approximation error
    F_norm = np.linalg.norm(LD - LD_lr - LD_b)
    logging.info("Using (B,R) = ({},{}), the approximated \% of LD: {}%".format(B, R, 100 - F_norm / LD_norm * 100))

    # use SVD on U to obtain optimal eigs (instead of eig-decomp UU')
    u, s, vh = np.linalg.svd(U, full_matrices = False, compute_uv = True)
    eigval = (s**2)[:R]
    eigvec = u[:, :R]

    # save approximation to files
    logging.info("Saving the LD approximation using (B,R) = ({},{}) to files.".format(B, R))
    output_fp = args.output_fp + "_B_{}_R_{}_{}_LRdecomp".format(B, R, args.LD_approx_method)
    np.savez(output_fp, w=eigval, v=eigvec, LD_b=LD_b)

    # set the number of experimetns to use for each setting
    num_valid_exp = min(int(args.LD_approx_num_valid_exp), df_valid_ss.shape[1])
    logging.info("Start validating the setting using {} experiments".format(num_valid_exp))

    # randomize validation samples
    valid_index = np.random.choice(df_valid_ss.shape[1], size=num_valid_exp, replace=False)

    # calculate HEELS estimates
    h2_est_arr = list()

    # initialize values (same across experiments - worse case scenario)
    sigma_g_0 = np.random.uniform(size=1)[0]
    sigma_e_0 = 1 - sigma_g_0

    for j in range(num_valid_exp):
        try:
            sigma_g, sigma_e = run_HEELS.HEELS_band_lr(args, df_valid_ss[["Z"+str(valid_index[j])]].values.reshape(-1,1), heels_format.band_format(LD_b, B), eigvec, eigval, sigma_g_0, sigma_e_0, m, n, args.YtY, tol=1e-3, maxIter=20)
            h2_est = np.real(sigma_g) / (np.real(sigma_g) + np.real(sigma_e))
            h2_est_arr.append(h2_est)

        except Exception as e:
            # logging.error(e, exc_info=True)
            logging.info('Unsuccessful convergence for validation experiment {}'.format(j+1))
            logging.info('Moving on to the next experiment.')

    # gather the statistics of h2 from experiments
    h2_est_arr = np.asarray(h2_est_arr)
    h2_est = np.nanmean(h2_est_arr)
    h2_bias = np.real(h2_est) - true_h2
    h2_sd = math.sqrt(np.var(h2_est_arr))
    h2_mse = math.sqrt(h2_bias**2 + h2_sd**2)

    logging.info("Total number of successfully converged validation experiments: {}".format(h2_est_arr.shape[0]))
    logging.info("Using (B,R) = ({},{}), the CV bias is {}.".format(B, R, h2_bias))
    logging.info("Using (B,R) = ({},{}), the CV SD is {}.".format(B, R, h2_sd))
    logging.info("Using (B,R) = ({},{}), the CV MSE is {}.".format(B, R, h2_mse))
    
    return F_norm, h2_bias, h2_sd, h2_mse, LD_b, eigval, eigvec

def CV_h2_seq(f_args):

    '''
    Running cross-validation to evaluate the performance of the "seq_band_lr" Banded + LR strategy. 

    CV statistics are calculated for the specified (B,R) hyperparameter setting.

    '''

    args, B, R, LD, LD_norm, m, n, true_h2, df_valid_ss = f_args

    LD_b = heels_format.dense_band(LD, B)

    if R != 0:
        eigval, eigvec = LR_func_optim(LD, LD_b, R, method = args.LR_decomp_method)
        LD_lr = eigvec.dot(np.diag(eigval).dot(np.transpose(eigvec)))

        # report intermediate approximation error
        F_norm = np.linalg.norm(LD - LD_lr - LD_b)
        logging.info("Using (B,R) = ({},{}), the approximated \% of LD: {}%".format(B, R, 100 - F_norm / LD_norm * 100))

        # save approximation to files
        logging.info("Saving the LD approximation using (B,R) = ({},{}) to files.".format(B, R))
        output_fp = args.output_fp + "_B_{}_R_{}_{}_LRdecomp".format(B, R, args.LD_approx_method)
        np.savez(output_fp, w=eigval, v=eigvec, LD_b=LD_b)
    else:
        logging.info("Effectively applying Band_only strategy since R = 0")
        # report intermediate approximation error
        F_norm = np.linalg.norm(LD - LD_b)
        logging.info("Using (B,R) = ({},{}), the approximated \% of LD: {}%".format(B, 0, 100 - F_norm / LD_norm * 100))

        # save approximation to files
        logging.info("Saving the LD approximation using (B,R) = ({},{}) to files.".format(B, 0))
        output_fp = args.output_fp + "_B_{}_R_{}_Band_only_LRdecomp".format(B, 0)
        np.savez(output_fp, LD_b=LD_b)

    # set the number of experimetns to use for each setting
    num_valid_exp = min(int(args.LD_approx_num_valid_exp), df_valid_ss.shape[1])
    logging.info("Start validating the setting using {} experiments".format(num_valid_exp))

    # randomize validation samples
    valid_index = np.random.choice(df_valid_ss.shape[1], size=num_valid_exp, replace=False)

    # calculate HEELS estimates
    h2_est_arr = list()

    # initialize values (same across experiments - worse case scenario)
    sigma_g_0 = np.random.uniform(size=1)[0]
    sigma_e_0 = 1 - sigma_g_0

    for j in range(num_valid_exp):
        try:
            if R != 0:
                sigma_g, sigma_e = run_HEELS.HEELS_band_lr(args, df_valid_ss[["Z"+str(valid_index[j])]].values.reshape(-1,1), heels_format.band_format(LD_b, B), eigvec, eigval, sigma_g_0, sigma_e_0, m, n, args.YtY, tol=1e-3, maxIter=20)
            else:
                sigma_g, sigma_e = run_HEELS.HEELS_band(args, df_valid_ss[["Z"+str(valid_index[j])]].values.reshape(-1,1), heels_format.band_format(LD_b, B), sigma_g_0, sigma_e_0, m, n, args.YtY, tol=1e-3, maxIter=20)

            h2_est = np.real(sigma_g) / (np.real(sigma_g) + np.real(sigma_e))
            h2_est_arr.append(h2_est)

        except Exception as e:
            # logging.error(e, exc_info=True)
            logging.info('Unsuccessful convergence for validation experiment {}'.format(j+1))
            logging.info('Moving on to the next experiment.')

    # gather the statistics of h2 from experiments
    h2_est_arr = np.asarray(h2_est_arr)
    h2_est = np.nanmean(h2_est_arr)
    h2_bias = np.real(h2_est) - true_h2
    h2_sd = math.sqrt(np.var(h2_est_arr))
    h2_mse = math.sqrt(h2_bias**2 + h2_sd**2)

    logging.info("Total number of successfully converged validation experiments: {}".format(h2_est_arr.shape[0]))
    logging.info("Using (B,R) = ({},{}), the CV bias is {}.".format(B, R, h2_bias))
    logging.info("Using (B,R) = ({},{}), the CV SD is {}.".format(B, R, h2_sd))
    logging.info("Using (B,R) = ({},{}), the CV MSE is {}.".format(B, R, h2_mse))
    
    if R == 0:
        logging.info("Fill in dummy zeros for the LR component, for compatibility with Band_only.")
        eigvec = np.zeros(shape=(m, 10))
        eigval = np.zeros(shape=(10,10))

    return F_norm, h2_bias, h2_sd, h2_mse, LD_b, eigval, eigvec

def CV_h2_psd_band(f_args):

    '''
    Running cross-validation to evaluate the performance of the "PSD_band_lr" Banded + LR strategy. 

    CV statistics are calculated for the specified (B,R) hyperparameter setting.

    '''

    args, B, R, LD, LD_norm, m, n, true_h2, df_valid_ss = f_args

    # set up objects for optim
    LD_b = heels_format.dense_band(LD, B)
    LD_b_norm = np.linalg.norm(LD_b)

    # initialize chol factor
    L0 = flatten_chol(np.random.rand(m, m), B)
    logging.info("Shape of the initial L0")
    
    # optimize for the banded part only
    # LD_banded = heels_format.band_format(LD_b, B)
    # L0 = PSD_adjust(B, LD_banded)
    # logging.info("size of L0: {}".format(L0.shape))
    # logging.info("size of x0: {}".format(np.asarray(L0).shape))
    opt = minimize(approx_err_banded, x0 = np.asarray(L0), args = (LD_b, B, LD_b_norm), jac = grd_U_banded, method = args.method, options = {'ftol': float(args.LD_approx_ftol)})

    # reconstruct optimized psd banded matrix
    L = rebuild_chol_mat(opt.x, m, B)
    LD_b = L.dot(np.transpose(L))

    eigval, eigvec = LR_func_optim(LD, LD_b, R, method = args.LR_decomp_method)
    LD_lr = eigvec.dot(np.diag(eigval).dot(np.transpose(eigvec)))

    # report intermediate approximation error
    F_norm = np.linalg.norm(LD - LD_lr - LD_b)
    logging.info("Using (B,R) = ({},{}), the approximated \% of LD: {}%".format(B, R, 100 - F_norm / LD_norm * 100))

    # save approximation to files
    logging.info("Saving the LD approximation using (B,R) = ({},{}) to files.".format(B, R))
    output_fp = args.output_fp + "_B_{}_R_{}_{}_LRdecomp".format(B, R, args.LD_approx_method)
    np.savez(output_fp, w=eigval, v=eigvec, LD_b=LD_b)

    # set the number of experimetns to use for each setting
    num_valid_exp = min(int(args.LD_approx_num_valid_exp), df_valid_ss.shape[1])
    logging.info("Start validating the setting using {} experiments".format(num_valid_exp))

    # randomize validation samples
    valid_index = np.random.choice(df_valid_ss.shape[1], size=num_valid_exp, replace=False)

    # calculate HEELS estimates
    h2_est_arr = list()

    # initialize values (same across experiments - worse case scenario)
    sigma_g_0 = np.random.uniform(size=1)[0]
    sigma_e_0 = 1 - sigma_g_0

    for j in range(num_valid_exp):
        try:
            sigma_g, sigma_e = run_HEELS.HEELS_band_lr(args, df_valid_ss[["Z"+str(valid_index[j])]].values.reshape(-1,1), heels_format.band_format(LD_b, B), eigvec, eigval, sigma_g_0, sigma_e_0, m, n, args.YtY, tol=1e-3, maxIter=20)
            h2_est = np.real(sigma_g) / (np.real(sigma_g) + np.real(sigma_e))
            h2_est_arr.append(h2_est)

        except Exception as e:
            # logging.error(e, exc_info=True)
            logging.info('Unsuccessful convergence for validation experiment {}'.format(j+1))
            logging.info('Moving on to the next experiment.')

    # gather the statistics of h2 from experiments
    h2_est_arr = np.asarray(h2_est_arr)
    h2_est = np.nanmean(h2_est_arr)
    h2_bias = np.real(h2_est) - true_h2
    h2_sd = math.sqrt(np.var(h2_est_arr))
    h2_mse = math.sqrt(h2_bias**2 + h2_sd**2)

    logging.info("Total number of successfully converged validation experiments: {}".format(h2_est_arr.shape[0]))
    logging.info("Using (B,R) = ({},{}), the CV bias is {}.".format(B, R, h2_bias))
    logging.info("Using (B,R) = ({},{}), the CV SD is {}.".format(B, R, h2_sd))
    logging.info("Using (B,R) = ({},{}), the CV MSE is {}.".format(B, R, h2_mse))
    
    return F_norm, h2_bias, h2_sd, h2_mse, LD_b, eigval, eigvec

def CV_h2_B_optim_R(f_args):

    '''
    Running cross-validation to evaluate the performance of the "seq_band_lr" or "PSD_band_lr" strategies.

    CV statistics are calculated for the specified B.
    The optimal R value is calculated within this function via Inc_SVD.

    '''

    args, B, LD, LD_norm, m, n, tol_R, true_h2, df_valid_ss = f_args
    
    # find the optimal approximation
    logging.info("Start optimization with B = {}, optimizing R using IncSVD".format(B))
    R_opt, eigvec, eigval, LD_lr, LD_b = optim_R(args, LD, B, tol_R)
    logging.info("Optimized R value is {}".format(R_opt))

    # evaluate the performance of approx
    F_norm = np.linalg.norm(LD - LD_lr - LD_b)
    logging.info("Using (B,R) = ({},{}), the approximated \% of LD: {}%".format(B, R_opt, 100 - F_norm / LD_norm * 100))

    # save approximation to files
    logging.info("Saving the LD approximation using (B,R) = ({},{}) to files.".format(B, R_opt))
    output_fp = args.output_fp + "_B_{}_R_{}_{}_LRdecomp".format(B, R_opt, args.LD_approx_method)
    np.savez(output_fp, w=eigval, v=eigvec, LD_b=LD_b)

    # set the number of experimetns to use for each setting
    num_valid_exp = min(int(args.LD_approx_num_valid_exp), df_valid_ss.shape[1])
    logging.info("Using {} experiments to validate each setting".format(num_valid_exp))

    # calculate HEELS estimates
    h2_est_arr = list()

    # initialize values (same across experiments - worse case scenario)
    sigma_g_0 = np.random.uniform(size=1)[0]
    sigma_e_0 = 1 - sigma_g_0

    for j in range(num_valid_exp):
        try:
            sigma_g, sigma_e = run_HEELS.HEELS_band_lr(args, df_valid_ss[["Z"+str(valid_index[j])]].values.reshape(-1,1), heels_format.band_format(LD_b, B), eigvec, eigval, sigma_g_0, sigma_e_0, m, n, args.YtY, tol=1e-3, maxIter=20)
            h2_est = np.real(sigma_g) / (np.real(sigma_g) + np.real(sigma_e))
            h2_est_arr.append(h2_est)

        except Exception as e:
            # logging.error(e, exc_info=True)
            logging.info('Unsuccessful convergence for validation experiment {}'.format(j+1))
            logging.info('Moving on to the next experiment.')

    # gather the statistics of h2 from experiments
    h2_est_arr = np.asarray(h2_est_arr)
    h2_est = np.nanmean(h2_est_arr)
    h2_bias = np.real(h2_est) - true_h2
    h2_sd = math.sqrt(np.var(h2_est_arr))
    h2_mse = math.sqrt(h2_bias**2 + h2_sd**2)

    logging.info("Total number of successfully converged validation experiments: {}".format(h2_est_arr.shape[0]))
    logging.info("Using (B,R) = ({},{}), the CV bias is {}.".format(B, R_opt, h2_bias))
    logging.info("Using (B,R) = ({},{}), the CV SD is {}.".format(B, R_opt, h2_sd))
    logging.info("Using (B,R) = ({},{}), the CV MSE is {}.".format(B, R_opt, h2_mse))
    
    return F_norm, h2_bias, h2_sd, h2_mse, LD_b, eigval, eigvec, R_opt

# Incremental SVD related 
def optim_R(args, LD, B, tol_R):

    '''
    Find the optimal low-rank component, given a pre-specified bandwidth,
    using the incremental SVD algorithm.
    Applicable only to sequential Band + LR strategies.

    
    Parameters:
    ------------
    LD: original full LD matrix
    B: pre-specified bandwidth
    tol_R: tolerance for LD approximation error, as the ratio of two matrix norms

    Returns: 
    ---------
    optimal number of low-rank factors
    components of the approximation: LD_lr, LD_b
    '''

    # Step 1: banded component
    if args.LD_approx_method == "seq_band_lr":
        LD_b = heels_format.dense_band(LD, B)
        
    elif args.LD_approx_method == "PSD_band_lr":
        LD_b = heels_format.dense_band(LD, B)
        LD_b_norm = np.linalg.norm(LD_b)

        # initialize chol factor
        m = LD_b.shape[0]
        L0 = flatten_chol(np.random.rand(m, m), B)
        opt = minimize(approx_err_banded, x0 = np.asarray(L0), args = (LD_b, B, LD_b_norm), jac = grd_U_banded, method = args.method, options = {'ftol': float(args.LD_approx_ftol)})

        # reconstruct optimized psd banded matrix
        L = rebuild_chol_mat(opt.x, m, B)
        LD_b = L.dot(np.transpose(L))
        logging.info("Finishing PSD approximation of the banded component: B = {}".format(B))

    F_norm = np.linalg.norm(LD - LD_b)
    LD_norm = np.linalg.norm(LD)
    logging.info("Before Inc SVD, only using the banded component (B = {}), \% of LD: {}%".format(B, 100 - F_norm / LD_norm * 100))

    # Step 2: low-rank component
    LD_resid = LD - LD_b
    max_m = round(args.min_sparsity * LD.shape[0])
    R_opt, eigvec_cum, eigval_cum, LD_lr = Inc_SVD(LD_resid, args.R_step_size, tol_R, max_m, B)
    # LD_lr = eigvec_cum.dot(np.diag(eigval_cum).dot(np.transpose(eigvec_cum)))

    return R_opt, eigvec_cum, eigval_cum, LD_lr, LD_b

def Inc_SVD(LD_target, step_size, tol_R, max_m, B, norm_func = np.linalg.norm):
    '''
    Incremental SVD algorithm for finding the optimal low-rank component
    given solution to the banded component.

    Parameters:
    ------------
    LD_target: the residual matrix after banding. Initial target of approximation
    step_size: the size of R increments
    tol_R: tolerance for LD approximation error, as the ratio of two matrix norms (norm of the error / norm of the original)
    norm_func: matrix norm function (can be self-defined, to account for LD decay)

    Intermediate objs:
    ------------------
    LD_increment: increment of the low-rank component estimated from the current iteration
    *accum: working approximating objects, low-rank component, eigenvectors and values
    approx_prop*: approximated proportion of a matrix, measured in norm_func

    Returns:
    ---------- 
    optimal number of low-rank factors
    eigenvectors and eigenvalues

    '''
    err = 1; i = 0; R = 0
    m = LD_target.shape[0]
    target_norm = norm_func(LD_target)

    LD_increment = np.zeros_like(LD_target)
    LD_accum = np.zeros_like(LD_target)
    eigvec_cum = np.zeros(shape=(m, step_size))
    eigval_cum = np.zeros(shape=(step_size, ))

    logging.info("Starting the Inc SVD algorithm with maximum R of {}".format(max_m))
    
    while abs(err) > tol_R and R <= max_m:
        i = i +1
        R = R + step_size
        logging.info("Inc SVD: performing low-rank decomposition with {} factors".format(R))
        # step_size = step_size / pow(2, i-1) # triming of step size for finer tuning
        eigval, eigvec = LR_func_optim(LD_target, LD_accum, step_size, method = "optim")

        # update the cumulative obj
        LD_increment = eigvec.dot(np.diag(eigval).dot(np.transpose(eigvec)))
        LD_accum = LD_accum + LD_increment
        eigvec_cum = np.hstack((eigvec_cum, eigvec))
        eigval_cum = np.append(eigval_cum, eigval, axis = 0)

        # report approximation performance
        logging.info("With (B,R) = ({},{}):".format(B, R))
        approx_prop_inc = norm_func(LD_increment) / norm_func(LD_target - LD_accum)
        logging.info("Approximated proportion \% of the working residual: {}%".format(approx_prop_inc*100))

        approx_prop = norm_func(LD_accum) / target_norm
        logging.info("Approximated proportion \% of the original off-banded LD: {}%".format(approx_prop*100))

        err = 1 - approx_prop
        logging.info("Overall error from Inc SVD: {}".format(err))

    return R, eigvec_cum, eigval_cum, LD_accum

# Main decomposition functions
def LDdecomp_auto(args, LD, norm_func = np.linalg.norm):
    '''
    Algorithm 1 in the paper.
    Sparse representation with hyperparameter tuning.

    If validation datasets is not provided, file path to raw genotype should be provided and the validation sumstats will be generated.

    Available decomposition strategies: (Banded + LR models)
    seq, joint, PSD_band. 

    Parameters
    -----------
    LD: original full LD matrix
    norm_func: matrix norm evaluating function for stopping criterion

    Save to files
    -------------
    Approximating objects
    Tuned hyperparameters: B, R
    Z-scores of the validation datasets, if not provided.

    Returns
    --------
    LD_dict: dictionary of approximating objects
    F_norm: norm of the approximation error, evaluated by "norm_func"

    '''
    logging.info("Start hyperparameter tuning process")
    logging.info("with starting values of (B,R): ({},{})".format(args.LD_approx_B, args.LD_approx_R))
    logging.info("Step size for increasing B: {}".format(args.B_step_size))
    logging.info("Step size for increasing R: {}".format(args.R_step_size))
    logging.info("Tolerance of CV bias is: {}".format(args.LD_approx_valid_bias_tol))

    num_valid_exp = int(args.LD_approx_num_valid_exp)
    logging.info("Number of validation experiments: {}".format(num_valid_exp))

    true_h2 = float(args.LD_approx_valid_h2)
    logging.info("True heritability (oracle): {}".format(true_h2))

    LD_norm = norm_func(LD)
    m_ref = LD.shape[0]
    n_ref = int(args.N)
    min_sparsity = args.min_sparsity

    # stopping criteria & reference
    tol_R = float(args.LD_approx_err_tol)
    max_m = round(min_sparsity * LD.shape[0])

    # set up candidate values of B
    B_grid = np.arange(start = round(args.LD_approx_B / 10)*10, stop = max_m, step = args.B_step_size)
    logging.info("Grid of B values:")
    logging.info(B_grid)

    logging.info(short_borderline)

    if args.LD_approx_valid_sumstats_fp is not None:
        # reading simulated pheno (rsID, N, Z1, Z2...)
        df_valid_ss = pd.read_csv(args.LD_approx_valid_sumstats_fp, delim_whitespace=True, index_col=None)
        assert df_valid_ss.shape[0] == m_ref, "Validation sumstats does not have the same number of SNPs as LD matrix"
        logging.info("Finish reading the validation sumstats")

        num_valid_exp = min(num_valid_exp, df_valid_ss.shape[1])

    else:
        # check if genotypes are provided
        if args.std_geno_fp is None and args.plink_bfile_fp is None:
            raise ValueError("Check --LD_approx_valid_sumstats_fp, --std_geno_fp, --plink_bfile_fp. If automtic LD decomposition is specified, they cannot be all None.")

        else :
            logging.info("No file path is specified in --LD_approx_valid_sumstats_fp. ")
            logging.info("Generating new validation sumstats from genotypes. ")

            if args.std_geno_fp is not None:
                logging.info("Reading genotypic files in matrix format from path: ")
                logging.info(args.std_geno_fp)
                geno_obj = np.load(args.std_geno_fp)
                std_geno_mat = geno_obj['std_geno']
                assert std_geno_mat.shape[0] == n_ref, "Mismatched number of individuals between bfile and LD input."
                assert std_geno_mat.shape[1] == m_ref, "Mismatched number of variants between bfile and LD input."
            else:
                logging.info("Reading genotypic files in PLINK binary format from path: ")
                logging.info(args.plink_bfile_fp)
                (bim, fam, bed) = read_plink(args.plink_bfile_fp)
                assert fam.shape[0] == n_ref, "Mismatched number of individuals between bfile and LD input."
                assert bim.shape[0] == m_ref, "Mismatched number of variants between bfile and LD input."

                # standardized genotype
                logging.info("Standardizing genoype matrix")
                geno_mat = bed.compute().T # p x n matrix
                geno_avg = np.nanmean(geno_mat, axis = 0)

                # fill in missing values with the avg
                nanidx = np.where(np.isnan(geno_mat))
                geno_mat[nanidx] = geno_avg[nanidx[1]]
                geno_sd = np.nanstd(geno_mat, axis = 0)
                # geno_mat[np.isnan(geno_mat)] = 0
                std_geno_mat = (geno_mat - geno_avg) / geno_sd

            logging.info("Completed reading in genotypic files for synthesizing Z-scores.")
            logging.info(short_borderline)

            # only genetic architecture enabled: GCTA model
            sigma2_arr = true_h2 / m_ref
            pheno_mat = np.full((n_ref, num_valid_exp), np.nan)
            var_vec = np.full((num_valid_exp,), np.nan)
            ols_mat = np.full((m_ref, num_valid_exp), np.nan)
            beta = np.random.normal(scale = np.power(sigma2_arr, 0.5), size = m_ref)

            for i in range(num_valid_exp):
                e = np.random.normal(scale = math.sqrt((1-true_h2)*float(args.YtY)), size = n_ref)
                y = std_geno_mat.dot(beta) + e
                var_vec[i] = np.var(y)
                ols_mat[:, i] = (np.dot(std_geno_mat.T, y) / math.sqrt(m_ref)).reshape(-1)

                # save validation files
                df_valid_ss = pd.DataFrame(data = ols_mat, columns=["Z"+str(i) for i in range(1, num_valid_exp + 1)])
                df_valid_ss.to_csv(args.output_fp + ".Z_ols", index=False, header=True, sep='\t')
                np.savetxt(args.output_fp + ".var", var_vec, delimiter = '\t')

    logging.info("Completed preparing for the validation summary statistics.")
    logging.info(short_borderline)


    # run the approximation or decomposition
    if args.LD_approx_method == "joint":

        # set up empty objects for final comparison
        B_hist = R_hist = F_norm_hist = LD_b_hist = eigval_hist = eigvec_hist = bias_hist = SD_hist = mse_hist = error_hist = []

        # (static) set of R values
        R_grid = np.arange(start = args.LD_approx_R, stop = max_m, step = args.R_step_size)
        logging.info("Grid of R values:")
        logging.info(R_grid)

        # initialize args.R with None to start anew
        args.LD_approx_R = None

        # start the cross-validation (outer layer: B; inner layer: R)
        i_B = 0
        while args.LD_approx_R is None and i_B <= (len(B_grid) - 1):
            B = B_grid[i_B]
            logging.info("Current value of B: {}".format(B))

            # parallel jobs across diff R values
            arg_list_ss = [(args, B, R_grid[j], LD, LD_norm, m_ref, n_ref, true_h2, df_valid_ss) for j in range(len(R_grid))]
            
            CV_h2_output =  joblib.Parallel(n_jobs = -1,
                                  backend='multiprocessing',
                                  verbose=1,
                                  batch_size=1)(joblib.delayed(CV_h2_joint)(f_args) for f_args in arg_list_ss)

            F_norm_l, h2_bias_l, h2_SD_l, h2_mse_l, LD_b_l, eigval_l, eigvec_l = zip(*CV_h2_output)

            # check if the criterion has been met (NOTE: always based on bias)
            R_criteria_index = np.where(np.array(np.abs(h2_bias_l)).reshape(-1,1) < float(args.LD_approx_valid_bias_tol))[0]

            # select (B,R) as soon as the criterion has been met; increase B if not
            if len(R_criteria_index) > 0:
                # if criterion has met, select the setting with the smallest bias or MSE
                selection_dict = {'bias': h2_bias_l, 'mse': h2_mse_l}
                if args.CV_metric is None:
                    raise ValueError("CV requires specifying --CV_metric, either using bias or mse.")
                else:
                    R_min_index = np.nanargmin(np.array(np.abs(selection_dict[args.CV_metric])))
                    args.LD_approx_R = R_grid[R_min_index]
                    args.LD_approx_B = B
                    F_norm = F_norm_l[R_min_index]
                    LD_b = LD_b_l[R_min_index]
                    eigval = eigval_l[R_min_index]
                    eigvec = eigvec_l[R_min_index]

                    logging.info("Based on {}, optimal values of (B,R): ({},{})".format(args.CV_metric, args.LD_approx_B, args.LD_approx_R))
                    logging.info("Statistics: Bias -- {}; SD -- {}; MSE -- {}".format(h2_bias_l[R_min_index], h2_SD_l[R_min_index], h2_mse_l[R_min_index]))
                break

            else:
                # add the current list of output to the history
                B_hist = B_hist + ([B]*len(R_grid)) # padding value
                R_hist = R_hist + list(R_grid)
                bias_hist = bias_hist + list(h2_bias_l)
                SD_hist = SD_hist + list(h2_SD_l)
                mse_hist = mse_hist + list(h2_mse_l)
                F_norm_hist = F_norm_hist + list(F_norm_l)
                LD_b_hist = LD_b_hist + list(LD_b_l)
                eigval_hist = eigval_hist + list(eigval_l)
                eigvec_hist = eigvec_hist + list(eigvec_l)

                # if criterion is not met, raise the B value
                logging.info("No (B,R) combo meets the criterion -> Increasing B by {}.".format(args.B_step_size))
                i_B = i_B + 1
                
        # If both B and R have both reached the threshold set by min_sparsity, but the CV bias criterion is still yet not met, select the setting with minimum bias
        if args.LD_approx_R is None:
            logging.info("No (B,R) values were found that led to CV bias lower than {}".format(float(args.LD_approx_valid_bias_tol)))
            logging.info("Selecting the best-performing setting possible.")

            selection_dict = {'bias': bias_hist, 'mse': mse_hist}

            if args.CV_metric is None:
                raise ValueError("CV requires specifying --CV_metric, either using bias or mse.")
            else:
                min_index = np.nanargmin(np.array(np.abs(selection_dict[args.CV_metric])))
                args.LD_approx_R = R_hist[min_index]
                args.LD_approx_B = B_hist[min_index]
                F_norm = F_norm_hist[min_index]
                LD_b = LD_b_hist[min_index]
                eigval = eigval_hist[min_index]
                eigvec = eigvec_hist[min_index]

                logging.info("Based on {}, optimal values of (B,R): ({},{})".format(args.CV_metric, args.LD_approx_B, args.LD_approx_R))
                logging.info("Statistics: Bias -- {}; SD -- {}; MSE -- {}".format(bias_hist[min_index], SD_hist[min_index], mse_hist[min_index]))

    else: # seq or PSD_band
        if args.LD_approx_inc_SVD:
            logging.info("Incremental SVD is used to optimize R values.")
            logging.info("CV is only applied to the optimal R value for each B.")
            arg_list_ss = [(args, B_grid[i], LD, LD_norm, m_ref, n_ref, tol_R, true_h2, df_valid_ss) for i in range(len(B_grid))]

            CV_h2_output =  joblib.Parallel(n_jobs = -1,
                                          backend='multiprocessing',
                                          verbose=1,
                                          batch_size=1)(joblib.delayed(CV_h2_B_optim_R)(f_args) for f_args in arg_list_ss)


            F_norm_l, h2_bias_l, h2_SD_l, h2_mse_l, LD_b_l, eigval_l, eigvec_l, R_opt_l = zip(*CV_h2_output)
            
            if args.CV_metric == "bias":
                # selection based on bias
                B_opt_index = np.nanargmin(np.array(np.abs(h2_bias_l)))
                args.LD_approx_B = B_grid[B_opt_index]
                args.LD_approx_R = R_opt_l[B_opt_index]
                logging.info("Based on bias, optimal values of (B,R): ({},{})".format(args.LD_approx_B, args.LD_approx_R))
            elif args.CV_metric == "mse":
                # selection based on MSE
                B_opt_index = np.nanargmin(np.array(np.abs(h2_mse_l)))
                args.LD_approx_B = B_grid[B_opt_index]
                args.LD_approx_R = R_opt_l[B_opt_index]
                logging.info("Based on MSE, optimal values of (B,R): ({},{})".format(args.LD_approx_B, args.LD_approx_R))
            else:
                raise ValueError("CV requires specifying --CV_metric, either using bias or MSE.")

            logging.info("Index of the optimal setting (B value): {}".format(B_opt_index))

            LD_b = LD_b_l[B_opt_index]
            eigval = eigval_l[B_opt_index]
            eigvec = eigvec_l[B_opt_index]
        else:
            logging.info("CV is applied to candidate (B,R) pairs on the search grid: ")
            logging.info("sequentially on B, parallel processes for R. ")

            # set up empty objects for final comparison
            B_hist = R_hist = F_norm_hist = LD_b_hist = eigval_hist = eigvec_hist = bias_hist = SD_hist = mse_hist = error_hist = []

            # (static) set of R values
            R_grid = np.arange(start = args.LD_approx_R, stop = max_m, step = args.R_step_size)
            logging.info("Grid of R values:")
            logging.info(R_grid)

            # initialize args.R with None to start anew
            args.LD_approx_R = None

            # start the cross-validation (outer layer: B; inner layer: R)
            i_B = 0
            logging.info(short_borderline)
    
            while args.LD_approx_R is None and i_B <= (len(B_grid)-1):
                B = B_grid[i_B]
                logging.info("Current value of B: {}".format(B))

                # parallel jobs across diff R values
                arg_list_ss = [(args, B, R_grid[j], LD, LD_norm, m_ref, n_ref, true_h2, df_valid_ss) for j in range(len(R_grid))]
                
                if args.LD_approx_method == "seq_band_lr":
                    CV_h2_func = CV_h2_seq
                elif args.LD_approx_method == "PSD_band_lr":
                    CV_h2_func = CV_h2_psd_band

                CV_h2_output =  joblib.Parallel(n_jobs = -1,
                                      backend='multiprocessing',
                                      verbose=1,
                                      batch_size=1)(joblib.delayed(CV_h2_func)(f_args) for f_args in arg_list_ss)

                F_norm_l, h2_bias_l, h2_SD_l, h2_mse_l, LD_b_l, eigval_l, eigvec_l = zip(*CV_h2_output)

                # check if the criterion has been met (NOTE: always based on bias)
                R_criteria_index = np.where(np.array(np.abs(h2_bias_l)).reshape(-1,1) < float(args.LD_approx_valid_bias_tol))[0]

                # select (B,R) as soon as the criterion has been met; increase B if not
                if len(R_criteria_index) > 0:
                    # if criterion has met, select the setting with the smallest bias or MSE
                    selection_dict = {'bias': h2_bias_l, 'mse': h2_mse_l}
                    if args.CV_metric is None:
                        raise ValueError("CV requires specifying --CV_metric, either using bias or mse.")
                    else:
                        R_min_index = np.nanargmin(np.array(np.abs(selection_dict[args.CV_metric])))
                        args.LD_approx_R = R_grid[R_min_index]
                        args.LD_approx_B = B
                        F_norm = F_norm_l[R_min_index]
                        LD_b = LD_b_l[R_min_index]
                        eigval = eigval_l[R_min_index]
                        eigvec = eigvec_l[R_min_index]

                        logging.info("Based on {}, optimal values of (B,R): ({},{})".format(args.CV_metric, args.LD_approx_B, args.LD_approx_R))
                        logging.info("Statistics: Bias -- {}; SD -- {}; MSE -- {}".format(h2_bias_l[R_min_index], h2_SD_l[R_min_index], h2_mse_l[R_min_index]))
                    break

                else:
                    # add the current list of output to the history
                    B_hist = B_hist + ([B]*len(R_grid)) # padding value
                    R_hist = R_hist + list(R_grid)
                    bias_hist = bias_hist + list(h2_bias_l)
                    SD_hist = SD_hist + list(h2_SD_l)
                    mse_hist = mse_hist + list(h2_mse_l)
                    F_norm_hist = F_norm_hist + list(F_norm_l)
                    LD_b_hist = LD_b_hist + list(LD_b_l)
                    eigval_hist = eigval_hist + list(eigval_l)
                    eigvec_hist = eigvec_hist + list(eigvec_l)

                    # if criterion is not met, raise the B value
                    logging.info("No (B,R) combo meets the criterion -> Increasing B by {}.".format(args.B_step_size))
                    i_B = i_B + 1

                logging.info(short_borderline)
                    
            # If both B and R have both reached the threshold set by min_sparsity, but the CV bias criterion is still yet not met, select the setting with minimum bias or MSE
            if args.LD_approx_R is None:
                logging.info("No (B,R) values were found that led to CV bias lower than {}".format(float(args.LD_approx_valid_bias_tol)))
                logging.info("Selecting the best-performing setting possible.")

                selection_dict = {'bias': bias_hist, 'mse': mse_hist}

                if args.CV_metric is None:
                    raise ValueError("CV requires specifying --CV_metric, either using bias or mse.")
                else:
                    min_index = np.nanargmin(np.array(np.abs(selection_dict[args.CV_metric])))
                    args.LD_approx_R = R_hist[min_index]
                    args.LD_approx_B = B_hist[min_index]
                    F_norm = F_norm_hist[min_index]
                    LD_b = LD_b_hist[min_index]
                    eigval = eigval_hist[min_index]
                    eigvec = eigvec_hist[min_index]

                    logging.info("Based on {}, optimal values of (B,R): ({},{})".format(args.CV_metric, args.LD_approx_B, args.LD_approx_R))
                    logging.info("Statistics: Bias -- {}; SD -- {}; MSE -- {}".format(bias_hist[min_index], SD_hist[min_index], mse_hist[min_index]))

    # below is applicable to all three Banded + LR strategies
    # ------------------------------------------------------------
    LD_lr = eigvec.dot(np.diag(eigval).dot(np.transpose(eigvec)))

    # compute norm of the err
    F_norm = norm_func(LD - LD_lr - LD_b)

    # save the eigen objects to file
    output_fp = args.output_fp + "_{}_LRdecomp".format(args.LD_approx_method)
    np.savez(output_fp, w=eigval, v=eigvec, LD_b=LD_b)

    # save selected (B, R) values
    with open(output_fp + ".opt_b", 'w') as f:
        f.write(str(args.LD_approx_B))
    with open(output_fp + ".opt_r", 'w') as f:
        f.write(str(args.LD_approx_R))

    LD_dict = dict({'LD_banded': heels_format.band_format(LD_b, args.LD_approx_B), 'eigvec': eigvec, 'eigval': eigval})

    return LD_dict, F_norm

def LDdecomp(args, LD, norm_func = np.linalg.norm):
    '''
    Sparse approximation of the LD. One-time decomposition, using pre-specified B,R.

    Parameters
    -----------
    LD: original full LD matrix
    min_sparsity: lowest sparsity allowed for LD approximation -- this sets the largest value of B and R for the sparse representation

    Available options: 
    -- Banded + LR models
    -- Spike coveriance models
    -- Existing methods (band only, LR only)

    Returns
    --------
    LD_dict: dictionary of approximating objects
    F_norm: norm of the approximation error, evaluated by "norm_func"

    '''
    logging.info("Start sparse matrix decomposition, \nusing pre-specified value of (B,R) = ({},{})".format(args.LD_approx_B, args.LD_approx_R))

    LD_norm = norm_func(LD)
    m_ref = LD.shape[0]


    #====================
    # Banded + LR models
    #====================
    if args.LD_approx_method == "joint":
        logging.info("Performing one-time decomposition using the joint method.")

        # initialize chol factors
        np.random.seed(7621)
        U0 = np.random.rand(m_ref, args.LD_approx_R)
        L0 = flatten_chol(np.random.rand(m_ref, m_ref), args.LD_approx_B)

        # initialize chol factors (non-randomly)
        # LD_banded = heels_format.band_format(LD, args.LD_approx_B)
        # L0 = PSD_adjust(args.LD_approx_B, LD_banded)

        # jointly optimize the banded and LR components
        opt = minimize(approx_err, x0 = np.asarray(tuple(U0.flatten()) + L0), args = (LD, args.LD_approx_R, args.LD_approx_B, LD_norm), jac = grd_U, method = args.method, options = {'ftol': float(args.LD_approx_ftol)})

        # read optimal results
        U = rebuild_U_mat(opt.x[:(m_ref*args.LD_approx_R)], m_ref, args.LD_approx_R)
        L = rebuild_chol_mat(opt.x[(m_ref*args.LD_approx_R):], m_ref, args.LD_approx_B)
        LD_b = L.dot(np.transpose(L))
        LD_lr = U.dot(np.transpose(U))

        # use SVD on U to obtain optimal eigs (as opposed to eig-decomp UU', which is more costly)
        u, s, vh = np.linalg.svd(U, full_matrices = False, compute_uv = True)
        eigval = (s**2)[:args.LD_approx_R]
        eigvec = u[:, :args.LD_approx_R]

        # compute norm of the err
        F_norm = norm_func(LD - LD_lr - LD_b)

    elif args.LD_approx_method == "seq_band_lr":
        logging.info("Performing one-time decomposition by sequentially banding and doing low-rank decomposition.")
        LD_b = heels_format.dense_band(LD, args.LD_approx_B)
        eigval, eigvec = LR_func_optim(LD, LD_b, args.LD_approx_R, method = args.LR_decomp_method)
        LD_lr = eigvec.dot(np.diag(eigval).dot(np.transpose(eigvec)))

        # compute norm of the err
        F_norm = norm_func(LD - LD_lr - LD_b)

    elif args.LD_approx_method == "seq_lr_band":
        logging.info("Performing one-time decomposition by sequentially doing low-rank decomposition and banding.")
        eigval, eigvec = LR_func_optim(LD, 0, args.LD_approx_R, method = args.LR_decomp_method)
        LD_lr = eigvec.dot(np.diag(eigval).dot(np.transpose(eigvec)))
        LD_delta = LD - LD_lr
        LD_b = heels_format.dense_band(LD_delta, args.LD_approx_B)

        # compute norm of the err
        F_norm = norm_func(LD - LD_lr - LD_b)

    elif args.LD_approx_method == "PSD_band_lr":
        logging.info("Performing one-time decomposition by 1) PSD approx the banded component and 2) doing low-rank decomposition on the residual.")

        LD_b = heels_format.dense_band(LD, args.LD_approx_B)
        LD_b_norm = norm_func(LD_b)
        np.random.seed(7621)
        L0 = flatten_chol(np.random.rand(m_ref, m_ref), args.LD_approx_B)

        # initialize chol factors (non-randomly)
        # LD_banded = heels_format.band_format(LD_b, args.LD_approx_B)
        # L0 = PSD_adjust(args.LD_approx_B, LD_banded)

        # optimize for the banded part only
        opt = minimize(approx_err_banded, x0 = np.asarray(L0), args = (LD_b, args.LD_approx_B, LD_b_norm), jac = grd_U_banded, method = args.method, options = {'ftol': float(args.LD_approx_ftol)})

        # reconstruct optimized banded matrix, with PSD guarantee
        L = rebuild_chol_mat(opt.x, m_ref, args.LD_approx_B)
        LD_b = L.dot(np.transpose(L))

        # low-rank decompose the residual
        eigval, eigvec = LR_func_optim(LD, LD_b, args.LD_approx_R, method = args.LR_decomp_method)
        LD_lr = eigvec.dot(np.diag(eigval).dot(np.transpose(eigvec)))

        # compute norm of the err
        F_norm = norm_func(LD - LD_lr - LD_b)

    elif args.LD_approx_method == "lr_PSD_band":
        logging.info("Performing one-time decomposition by 1) doing low-rank decomposition on the residual and 2) PSD approx the banded component.")
        
        eigval, eigvec = LR_func_optim(LD, 0, args.LD_approx_R, method = args.LR_decomp_method)
        LD_lr = eigvec.dot(np.diag(eigval).dot(np.transpose(eigvec)))
        LD_delta = LD - LD_lr

        LD_b = heels_format.dense_band(LD_delta, args.LD_approx_B)
        LD_b_norm = norm_func(LD_b)
        np.random.seed(7621)
        L0 = flatten_chol(np.random.rand(m_ref, m_ref), args.LD_approx_B)

        # initialize chol factors (non-randomly)
        # LD_banded = heels_format.band_format(LD_b, args.LD_approx_B)
        # L0 = PSD_adjust(args.LD_approx_B, LD_banded)

        # optimize for the banded part only
        opt = minimize(approx_err_banded, x0 = np.asarray(L0), args = (LD_b, args.LD_approx_B, LD_b_norm), jac = grd_U_banded, method = args.method, options = {'ftol': float(args.LD_approx_ftol)})

        # reconstruct optimized banded matrix, with PSD guarantee
        L = rebuild_chol_mat(opt.x, m_ref, args.LD_approx_B)
        LD_b = L.dot(np.transpose(L))

        # compute norm of the err
        F_norm = norm_func(LD - LD_lr - LD_b)

    elif args.LD_approx_method == "iterative":
        # TO TEST: suspect the performance to be quite unstable
        # unused b/c computationally intensive and requires an arbitrary # of iterations
        num_iter = 5
        logging.info("Performing one-time decomposition by iteratively ({} times) updating the banded and low-rank part.".format(num_iter))

        # init LD_b
        LD_b = heels_format.dense_band(LD, args.LD_approx_B)
        
        for i in range(num_iter):
            w, v = LR_func(LD, LD_b, args.LD_approx_R)
            LD_b = Band_func(args, LD, w, v)
            LD_b = heels_format.unband_format(LD_b) # dense format for next iteration

        eigval = w[-args.LD_approx_R:][::-1]
        eigvec = v[:, -args.LD_approx_R:][:, ::-1]
        LD_lr = eigvec.dot(np.diag(eigval).dot(np.transpose(eigvec)))

        # compute norm of the err
        F_norm = norm_func(LD - LD_lr - LD_b)

    #===================
    # Spiked cov models
    #===================
    elif args.LD_approx_method == "spike_LR" or args.LD_approx_method == "spike_PCA":
        logging.info("Performing one-time decomposition using the spiked covariance model.")
        logging.info("Spike method: {}".format(args.LD_approx_method))
        logging.info("Only homo variance is possible.")

        m = LD.shape[0]; r = int(args.LD_approx_R)

        logging.info("Step 1: Pre-estimation of the residual sigma")

        # Separate between different sub-sampling strategies
        if args.resid_sigma_estMethod == 1:
            rr = round(r*1.5)
            logging.info("Using top sub-matrix of LD with size {}".format(rr))
            LD_pre = LD[:rr, :rr]
            w_pre, v_pre = eig_banded(heels_format.band_format(LD_pre, rr), lower = True)
            eigval = np.sort(w_pre)[::-1]

            # capping the diagonal element to be larger than 1
            resid_sigma = np.nanmax([1, np.nanmin(eigval)])
            # resid_sigma = np.nanmin(eigval)
        elif args.resid_sigma_estMethod == 2:
            rr = round(r*1.5)
            logging.info("Using sub-sampling of LD with size {}".format(rr))
            resid_sigma = est_resid_sigma_subsample(LD, rr, m)

        elif args.resid_sigma_estMethod == 3:
            rr = round(r*1.5)
            logging.info("Using the average from 10 sub-sampling with size {}".format(rr))
            resid_sigma_l = list()
            for i in range(10):
                resid_sigma_l.append(est_resid_sigma_subsample(LD, rr, m, seed = i))
            logging.info("Estimated residual sigma across 10 exp: ")
            logging.info(resid_sigma_l)
            resid_sigma = np.nanmean(resid_sigma_l)

        # dummy banded matrix with only the diagonal
        LD_b = np.eye(m)*resid_sigma

        # subtract residual from the diagonal
        LD_lr = LD - LD_b

        logging.info("Step 2: Decomposition of the LR component")

        if args.LD_approx_method == "spike_LR":
            # LR decompose the residual LD
            LD_banded = heels_format.band_format(LD_lr, LD_lr.shape[0])
            w, v = eig_banded(LD_banded, lower = True, select = 'i', select_range = (m-r, m-1))

            # reconstruct LD based on truncated SVD
            eigval = np.sort(w[-r:])[::-1]
            eigvec = v[:, -r:][:, ::-1]
            LD_lr = eigvec.dot(np.diag(eigval).dot(np.transpose(eigvec)))

        elif args.LD_approx_method == "spike_PCA":
            # # Determine the choice of p, p_0 and l based on Thm 4.3, 4.4 of Shen et al.
            # p = max(2*r, r+7)
            # p_0 = p+5
            # q = max(50, round(math.log(n / p_0) / math.log(p) + 5))
            # L = min(100, round(m * r / p))
            # p = r; p_0 = r

            # run PCA on the LR part
            # logging.info("Running fast PCA with params: ")
            # logging.info("p = {}, p-final = {}, L = {}, q = {}".format(p, p_0, L, q))
            # eigvec, eigval = fast_pca(LD_lr, r, p_sub = p, p_final = p_0, L = L, q = q)

            logging.info("Running fast PCA using randomized SVD")
            eigs = skmath.randomized_svd(delta_LD, n_components=r, n_oversamples=2*m-r)
            eigvec = eigs[0]
            eigval = eigs[1]

            # assembly LR part
            LD_lr = eigvec.dot(np.diag(eigval).dot(np.transpose(eigvec)))

        # report the accuracy of in-sample approx
        F_norm = norm_func(LD - LD_lr - LD_b)

    elif args.LD_approx_method == "spike_LR_joint":
        logging.info("Performing one-time decomposition using the spiked covariance model.")
        logging.info("Spike method: {}".format(args.LD_approx_method))

        m = LD.shape[0]; r = int(args.LD_approx_R)

        # initialize low-rank
        np.random.seed(7621)
        U0 = np.random.rand(m_ref, args.LD_approx_R)

        # initialize the diagonal element value
        if args.spike_LR_diag == "homo":
            resid_sigma0 = [np.random.uniform(size=1)[0]]
        elif args.spike_LR_diag == "hetero":
            resid_sigma0 = list(np.random.uniform(size=m))

        # jointly optimize the diagonal and LR matrices
        logging.info("Start optimization")
        opt = minimize(approx_err_spike_LR, x0 = np.asarray(list(U0.flatten()) + resid_sigma0), args = (LD, args.LD_approx_R, LD_norm, args.spike_LR_diag), jac = grd_U_spike_LR, method = args.method, options = {'ftol': float(args.LD_approx_ftol)})

        # read optimal results
        U = rebuild_U_mat(opt.x[:(m*args.LD_approx_R)], m, args.LD_approx_R)
        resid_sigma = opt.x[(m*args.LD_approx_R):]

        if args.spike_LR_diag == "homo":
            LD_b = np.eye(m) * resid_sigma
        elif args.spike_LR_diag == "hetero":
            LD_b = np.diag(resid_sigma)

        LD_lr = U.dot(np.transpose(U))

        # use SVD on U to obtain optimal eigs
        u, s, vh = np.linalg.svd(U, full_matrices = False, compute_uv = True)
        eigval = (s**2)[:args.LD_approx_R]
        eigvec = u[:, :args.LD_approx_R]

        # report the accuracy of in-sample approx
        F_norm = np.linalg.norm(LD - LD_lr - LD_b)

    #===================
    # Existing models
    #===================
    elif args.LD_approx_method == "Band_only":
        # banding only, no low-rank part
        logging.info("Performing one-time decomposition using banded matrix only.")
        LD_b = heels_format.dense_band(LD, args.LD_approx_B)
        F_norm = np.linalg.norm(LD - LD_b)

    elif args.LD_approx_method == "LR_only":
        # only apply TSVD to the LD matrix, no banding
        logging.info("Performing one-time decomposition using truncated SVD only.")
        R = int(args.LD_approx_R)
        eigval, eigvec = LR_func_optim(LD, 0, R, method = args.LR_decomp_method)
        LD_lr = eigvec.dot(np.diag(eigval).dot(np.transpose(eigvec)))
        F_norm = np.linalg.norm(LD - LD_lr)

    #====================
    # Saving optimized
    #====================
    logging.info("Saving sparse representation to files.")
    
    output_fp = args.output_fp + "_{}_LRdecomp".format(args.LD_approx_method)

    if args.LD_approx_method == "Band_only":
        np.savez(output_fp, LD_b=LD_b)
    elif args.LD_approx_method == "LR_only":
        np.savez(output_fp, w=eigval, v=eigvec)
    else:
        np.savez(output_fp, w=eigval, v=eigvec, LD_b=LD_b)

    logging.info(short_borderline)
    #======================
    # Consolidating approx
    #======================
    if args.LD_approx_method == "LR_only":
        LD_dict = dict({'eigvec': eigvec, 'eigval': eigval})
        m_ref = eigvec.shape[0]
    elif args.LD_approx_method == "Band_only":
        LD_dict = dict({'LD_banded': heels_format.band_format(LD_b, args.LD_approx_B)})
        m_ref = LD_b.shape[0]
    elif args.LD_approx_method is not None:
        LD_dict = dict({'LD_banded': heels_format.band_format(LD_b, args.LD_approx_B), 'eigvec': eigvec, 'eigval': eigval})
        m_ref = eigvec.shape[0]

    #===================
    # Thresholding
    #===================
    # threshold by a value
    if args.threshold_r is not None:
      LD[LD <= args.threshold_r] = 0

    # threshold by sparse (quantile of the values)

    return LD_dict, F_norm

