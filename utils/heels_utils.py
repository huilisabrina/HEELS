#!/usr/bin/env python

#-------------------------------------------------------
# Summary Statistics and Linkage Disequilibrium

# A collection of functions to be called independently
# Used for: simulations, collaborative work, etc.

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

def LR_decomp_Band_LR(B, R, LD):
    LD_b = dense_band(LD, B)
    w, v = LR_func(LD, LD_b, R)
    eigval = w[-R:][::-1]
    eigvec = v[:, -R:][:, ::-1]
    return LD_b, eigvec, eigval

def LD_select(i, LDsnp_index, block_bounds, LD_list):
    idx1 = np.arange(block_bounds[i+1] - block_bounds[i])
    idx2 = LDsnp_index[block_bounds[i]:block_bounds[i+1]]
    idx = idx1[idx2]
    logging.info(idx)
    logging.info(idx.shape)
    return LD_list[i][idx, idx]

def chol_update(L, x): # rank-one update to chol factors
    n = x.shape[0]
    for k in range(n):
        r = np.sqrt(L[k, k]**2 + x[k]**2)
        c = r / L[k, k]
        s = x[k] / L[k, k]
        L[k, k] = r
        if k < n:
            L[k+1:n, k] = (L[(k+1):n, k] + s * x[(k+1):n]) / c
            x[(k+1):n] = c * x[(k+1):n] - s * L[(k+1):n, k]
    return L

# Util functions for simulation (no args.)
def HEELS_banded_util(Z_m, LD_0, sigma_g_0, sigma_e_0, m, n, YtY = None, bandwidth = None, update_sigma_g = "Seq", constrain_sigma = False, tol=1e-3, maxIter=100):

    # to ensure input are consistent
    assert m == LD_0.shape[0], "Inconsistent dimension of LD_0 and m"

    # initialize algorithm
    sigma_g = sigma_g_0
    sigma_e = sigma_e_0
    sigma_g_list = [sigma_g]
    sigma_e_list = [sigma_e]
    diff_g = 100
    diff_e = 100

    if bandwidth is None:
        bandwidth = LD_0.shape[0]
    LD_banded = band_format(LD_0, bandwidth)

    i = 0

    while ((abs(diff_g) > tol or abs(diff_e) > tol) and (i < maxIter)):
        print("Iteration: {}".format(i))

        sigma_g_prev = sigma_g 
        sigma_e_prev = sigma_e
        lam = sigma_e / sigma_g

        # solve W_t^{-1}Z:
        print("Starting backsolve: ")

        # use cholesky of banded matrix for W^-1y and tr(W^-1)
        chol_start = time.time()
        W_t = LD_banded.copy()
        W_t[0,:] = W_t[0,:] + lam

        # solve W^{-1}Z:
        chol = cholesky_banded(W_t, lower = True)
        BLUP_t = cho_solve_banded((chol, True), Z_m)
        chol_time = round(time.time() - chol_start, 2)
        print("Cholesky time: {T}".format(T=chol_time))

        # solve (tr(W^{-1}))
        print("Starting trace-inverse: ")
        trInv_start = time.time()
        inv = cho_solve_banded((chol, True), np.eye(m))
        trace = np.trace(inv)
        trInv_time = round(time.time() - trInv_start, 2)
        print("Trace-inverse time: {T}".format(T=trInv_time))

        # update variance components
        print("Updating variance components: ")
        update_sig_start = time.time()

        # alternative updating equation:
        if update_sigma_g == "noSeq":
            sigma_g = np.real(np.matmul(np.transpose(BLUP_t), BLUP_t).item(0) / (m - trace))
        else:
            sigma_g = np.real(1/m*(np.matmul(np.transpose(BLUP_t), BLUP_t).item(0) + sigma_e*trace))

        sigma_e = np.real(1/n*(n - np.matmul(np.transpose(Z_m), BLUP_t).item(0)))

        if constrain_sigma:
            print("Rescaling updated values s.t. sigma_g^2 + sigma_e^2 = 1")
            sigma_tot = sigma_g + sigma_e
            sigma_g = sigma_g / sigma_tot
            sigma_e = sigma_e / sigma_tot
        elif YtY is not None: # if y'y is known:
            sigma_e = np.real(1/n*(YtY - np.matmul(np.transpose(Z_m), BLUP_t).item(0)))

        print("Checking if current values are completely out of the bounds")
        if sigma_g > 10 or sigma_e > 10:
            print("Invalid values - failure to converge")
            break

        # ==============
        # record results
        # ==============
        diff_g = sigma_g - sigma_g_prev
        diff_e = sigma_e - sigma_e_prev
        print("Current sigma_g: {} \n".format(sigma_g))
        print("Current sigma_e: {} \n".format(sigma_e))
        print("Difference in sigma_g: {} \n".format(diff_g))
        print("Difference in sigma_e: {} \n".format(diff_e))
        sigma_g_list.append(sigma_g)
        sigma_e_list.append(sigma_e)
        i = i + 1

    print('Number of iterations: {I}'.format(I=i))

    LD_banded = band_format(LD_0, bandwidth)
    eigs = eigvals_banded(LD_banded, lower = True)

    var_mat_ssld = HEELS_variance(sigma_g, sigma_e, eigs, n, m)
    se_g = np.sqrt(var_mat_ssld[1, 1])
    se_e = np.sqrt(var_mat_ssld[0, 0])

    # multi Delta function for var of h2
    grad_denom = (sigma_g + sigma_e)**2
    grad_vec = np.asarray([(-1)* sigma_g / grad_denom, sigma_e / grad_denom])
    se_h2 = np.sqrt(np.sum(np.multiply(np.einsum('i,ij->j', grad_vec, var_mat_ssld), grad_vec)))
    
    return sigma_g, sigma_e, se_g, se_e, se_h2

def HEELS_lowrank_util(Z_m, LD_approx_method, LD_dict, sigma_g_0, sigma_e_0, m, n, YtY = None, update_sigma_g = "Seq", constrain_sigma = False, tol=1e-3, maxIter=100):
    
    # initialize algorithm
    sigma_g = sigma_g_0
    sigma_e = sigma_e_0
    sigma_g_list = [sigma_g]
    sigma_e_list = [sigma_e]
    diff_g = 100
    diff_e = 100

    # timing of the iterative procedure
    i = 0

    while ((abs(diff_g) > tol or abs(diff_e) > tol) and (i < maxIter)):
        print("Iteration: {}".format(i))

        sigma_g_prev = sigma_g 
        sigma_e_prev = sigma_e
        lam = sigma_e / sigma_g

        # update the BLUP and compute trace
        if LD_approx_method == "LR_only":
            BLUP_t, trace = HEELS_lr_iter(lam, m, Z_m, LD_dict['eigvec'], LD_dict['eigval'])
        elif LD_approx_method == "Band_only":
            BLUP_t, trace = HEELS_band_iter(lam, m, Z_m, LD_dict['LD_banded'])
        elif LD_approx_method is not None:
            BLUP_t, trace = HEELS_band_lr_iter(lam, m, Z_m, LD_dict['LD_banded'], LD_dict['eigvec'], LD_dict['eigval'], LD_approx_method = LD_approx_method)
        else:
            BLUP_t, trace = HEELS_iter(lam, m, Z_m, LD_dict['LD_banded'], sigma_e, YtY)

        # alternative updating equation:
        if update_sigma_g == "noSeq":
            sigma_g = np.real(np.matmul(np.transpose(BLUP_t), BLUP_t).item(0) / (m - trace))
        else:
            sigma_g = np.real(1/m*(np.matmul(np.transpose(BLUP_t), BLUP_t).item(0) + sigma_e*trace))

        sigma_e = np.real(1/n*(n - np.matmul(np.transpose(Z_m), BLUP_t).item(0)))

        if constrain_sigma:
            print("Rescaling updated values s.t. sigma_g^2 + sigma_e^2 = 1")
            sigma_tot = sigma_g + sigma_e
            sigma_g = sigma_g / sigma_tot
            sigma_e = sigma_e / sigma_tot
        elif YtY is not None: # if y'y is known:
            sigma_e = np.real(1/n*(YtY - np.matmul(np.transpose(Z_m), BLUP_t).item(0)))

        print("Checking if current values are completely out of the bounds")

        if sigma_g > 10 or sigma_e > 10:
            print("Invalid values - failure to converge")
            break

        # ==============
        # record results
        # ==============
        diff_g = sigma_g - sigma_g_prev
        diff_e = sigma_e - sigma_e_prev
        print("Current sigma_g: {} \n".format(sigma_g))
        print("Current sigma_e: {} \n".format(sigma_e))
        print("Difference in sigma_g: {} \n".format(diff_g))
        print("Difference in sigma_e: {} \n".format(diff_e))
        sigma_g_list.append(sigma_g)
        sigma_e_list.append(sigma_e)
        i = i + 1

    # Compute variances using the approximating form of the LD
    # Current implementation only allows for Band + LR form
    var_mat_ssld = HEELS_variance_lowrank(sigma_g, sigma_e, LD_dict['LD_banded'], LD_dict['eigval'], LD_dict['eigvec'], n, m)
    se_g = np.sqrt(var_mat_ssld[1, 1])
    se_e = np.sqrt(var_mat_ssld[0, 0])

    # multi Delta function for var of h2
    grad_denom = (sigma_g + sigma_e)**2
    grad_vec = np.asarray([(-1)* sigma_g / grad_denom, sigma_e / grad_denom])
    se_h2 = np.sqrt(np.sum(np.multiply(np.einsum('i,ij->j', grad_vec, var_mat_ssld), grad_vec)))
    
    return sigma_g, sigma_e, se_g, se_e, se_h2

def HEELS_block_util(Z_m_list, LD_dict_list, sigma_g_0, sigma_e_0, m_list, n, YtY = None, update_sigma_g = "Seq", constrain_sigma = False, tol=1e-3, maxIter=100):

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
    print('HEELS procedure started at {T}'.format(T=time.ctime()))
    i = 0

    # block-specific
    m = np.sum(m_list)
    BLUP_t = np.zeros(shape=(m,))
    traces = np.zeros(shape=(m,))
    Z_index = np.insert(np.cumsum(np.asarray(m_list)), 0, 0)
    Z_m = np.hstack(Z_m_list)

    while ((abs(diff_g) > tol or abs(diff_e) > tol) and (i < maxIter)):
        print("Iteration: {}".format(i))

        sigma_g_prev = sigma_g 
        sigma_e_prev = sigma_e
        lam = sigma_e / sigma_g

        # update BLUP (this part can be potentially parallelized)
        for k in range(K):
            BLUP_t[Z_index[k]:Z_index[k+1]], traces[k] = block_BLUP_update_util(lam, m_list[k], Z_m_list[k], LD_dict_list[k])

        # aggregate block results
        trace = np.sum(traces)

        # adjustment related to updating sigma_g
        if update_sigma_g == "noSeq":
            sigma_g = np.matmul(np.transpose(BLUP_t), BLUP_t).item(0) / (m - trace)
        else:
            sigma_g = 1/m*(np.matmul(np.transpose(BLUP_t), BLUP_t).item(0) + sigma_e*trace)

        sigma_e = 1/n*(n - np.matmul(np.transpose(Z_m), BLUP_t).item(0))

        if constrain_sigma:
            print("Rescaling updated values s.t. sigma_g^2 + sigma_e^2 = 1")
            sigma_tot = sigma_g + sigma_e
            sigma_g = sigma_g / sigma_tot
            sigma_e = sigma_e / sigma_tot
        elif YtY is not None: # if y'y is known:
            sigma_e = np.real(1/n*(YtY - np.matmul(np.transpose(Z_m), BLUP_t).item(0)))

        print("Checking if current values are completely out of the bounds")

        if sigma_g > 10 or sigma_e > 10:
            print("Invalid values - failure to converge")
            break

        # ==============
        # record results
        # ==============
        diff_g = sigma_g - sigma_g_prev
        diff_e = sigma_e - sigma_e_prev
        print("Current sigma_g: {} \n".format(sigma_g))
        print("Current sigma_e: {} \n".format(sigma_e))
        print("Difference in sigma_g: {} \n".format(diff_g))
        print("Difference in sigma_e: {} \n".format(diff_e))
        sigma_g_list.append(sigma_g)
        sigma_e_list.append(sigma_e)
        i = i + 1

    print('Number of iterations: {I}'.format(I=i))

    # combine eigenvalues
    print("Length of LD dict and m_list: ")
    print(len(LD_dict_list))
    print(len(m_list))

    eigs_list = list()
    for k in range(K):
        LD_banded = LD_dict_list[k]['LD_banded']
        eigs = eigvals_banded(LD_banded, lower = True)
        eigs_list.append(eigs)
    eigs = np.vstack(eigs_list)
    print("Shape of the eigenvalues")
    print(eigs.shape)

    var_mat_ssld = HEELS_variance(sigma_g, sigma_e, eigs, n, m)
    se_g = np.sqrt(var_mat_ssld[1, 1])
    se_e = np.sqrt(var_mat_ssld[0, 0])

    # multi Delta function for var of h2
    grad_denom = (sigma_g + sigma_e)**2
    grad_vec = np.asarray([(-1)* sigma_g / grad_denom, sigma_e / grad_denom])
    se_h2 = np.sqrt(np.sum(np.multiply(np.einsum('i,ij->j', grad_vec, var_mat_ssld), grad_vec)))
    
    return sigma_g, sigma_e, se_g, se_e, se_h2

# Likelihood functions
def ML_util(X, y, sigma_g_0, sigma_e_0, method="BFGS", constraint=False):
    (N,M) = X.shape

    GRM = np.matmul(X, np.transpose(X))
    eigvals = np.linalg.eigvalsh(GRM)

    if constraint:
        bounds = Bounds([0, 0], [1.0, 1.0])
        opt = minimize(llfun, x0 = np.array([sigma_g_0, sigma_e_0]), args=(GRM, y, eigvals), method=method, bounds=bounds)
    else:
        opt = minimize(llfun, x0 = np.array([sigma_g_0, sigma_e_0]), args=(GRM, y, eigvals), method=method)

    sigma_g = opt.x[0]
    sigma_e = opt.x[1]
    print("ML solution of sigma_g: {}".format(sigma_g))
    print("ML solution of sigma_e: {}".format(sigma_e))

    # compute information matrix
    var_mat_mle = ML_variance(sigma_g, sigma_e, eigvals)
    se_g = np.sqrt(var_mat_mle[1, 1])
    se_e = np.sqrt(var_mat_mle[0, 0])

    # multi Delta function for var of h2
    grad_denom = (sigma_g + sigma_e)**2
    grad_vec = np.asarray([(-1)* sigma_g / grad_denom, sigma_e / grad_denom])
    se_h2 = np.sqrt(np.sum(np.multiply(np.einsum('i,ij->j', grad_vec, var_mat_mle), grad_vec)))

    return sigma_g, sigma_e, se_g, se_e, se_h2

def llfun(params, GRM, y, eigvals):
    sigma_g = params[0]
    sigma_e = params[1]
    lam = sigma_e / sigma_g
    N = GRM.shape[0]
    V_scaled = GRM + lam * np.eye(N)
    x = linalg.cg(V_scaled, y)
    assert x[1] == 0, "PCG does not converge"
    Vy_sol = x[0]
    zz = sigma_g*eigvals + sigma_e

    # only use positive eigenvalues
    ll = np.sum(np.log(zz[zz>0])) + np.matmul(np.transpose(y), Vy_sol) / sigma_g
    # logging.info("Likelihood: {}".format(ll))

    return ll

def Ufun(params, GRM, X, y, eigvals):
    sigma_g = params[0]
    sigma_e = params[1]
    lam = sigma_e / sigma_g
    N = GRM.shape[0]
    V_scaled = GRM + lam * np.eye(N)

    x = linalg.cg(V_scaled, y)
    assert x[1] == 0, "PCG does not converge"
    Vy_sol = x[0]

    zz = sigma_g*eigvals + sigma_e

    ll = np.sum(np.log(zz[zz>0])) + np.matmul(np.transpose(y), Vy_sol) / sigma_g

    V_inv = np.linalg.inv(V_scaled)
    Vinv_XXt = np.matmul(V_inv, GRM)
    Z_w = np.matmul(np.transpose(X), Vy_sol)

    Ug = np.trace(Vinv_XXt / sigma_g) - np.dot(Z_w, Z_w) / (sigma_g**2)
    Ue = np.trace(V_inv / sigma_g) - np.dot(Vy_sol, Vy_sol) / (sigma_g**2)

    return ll, Ug, Ue


# Variance estimator
def compute_logL(lam, m, n, Z_m, LD_0, sigma_e=None, YtY=1, null=False):

    '''
    Evaluate the likelihood, either under the null or under the alternative
    
    '''

    # solve W_t^{-1}Z:
    W_t = LD_0.copy()
    W_t[0,:] = W_t[0,:] + lam

    # solve W^{-1}Z:
    # since LD_0 naturally should be PSD
    chol = cholesky_banded(W_t, lower = True)
    BLUP_t = cho_solve_banded((chol, True), Z_m)

    # calculate log-likelihood from sumstats + LD
    # chol_decile = np.percentile(chol[0,:], np.arange(0, 100, 10))
    # logging.info(chol_decile)
    if null:
        Z_ll = (-1)*0.5*(YtY - np.sum(np.multiply(BLUP_t, Z_m)))
    else:
        Z_ll = (-1)*0.5*(n*np.log(sigma_e) + 2*np.sum(np.log(chol[0,:])) - m*np.log(lam) + ((YtY - np.sum(np.multiply(BLUP_t, Z_m)))/sigma_e))

    # the following does not work due to numerical overflow
    # det_W = np.prod(np.square(chol[0,:]) / lam)
    # Z_ll = (-1)*0.5*(n*np.log(sigma_e) + np.log(det_W) + ((YtY - np.sum(np.multiply(BLUP_t, Z_m)))/sigma_e))

    return Z_ll

def HEELS_variance(sigma_g, sigma_e, eigvals, n, m, eff_m=False, r=None):
    if r is None: #otherwise, only use the top eigenvalues
        r = eigvals.shape[0]
    lam = sigma_e / sigma_g
    tr_Winv = np.sum(1 / (eigvals[-r:] + lam))
    tr_Winv_2 = np.sum(1 / ((eigvals[-r:] + lam)**2))

    # logging.info("Used eigvals")
    # logging.info(eigvals[-r:])

    if eff_m:
        eff_m = np.sum(eigvals / (eigvals + lam))
        logging.info("The effective M is: {}".format(eff_m))

    i11 = (n - m) / (sigma_e**2) + tr_Winv_2 / (sigma_g**2)
    i12 = tr_Winv / (sigma_g**2) - (sigma_e*tr_Winv_2) / (sigma_g**3)
    i22 = m / (sigma_g**2) - (2 * sigma_e * tr_Winv) / (sigma_g**3) + (sigma_e**2 * tr_Winv_2) / (sigma_g**4)
    obInfo = 0.5*np.matrix([[i11, i12], [i12, i22]])
    var_mat = np.linalg.inv(obInfo)

    return var_mat

def HEELS_variance_new(sigma_g, sigma_e, eigvals, n, m, r=None, constrain_sigma=False):

    '''General variance formula for both the normalized and unnormalized cases.'''

    if r is None:
        r = eigvals.shape[0]
    lam = sigma_e / sigma_g
    tr_Winv = np.sum(1 / (eigvals[-r:] + lam))
    tr_Winv_2 = np.sum(1 / ((eigvals[-r:] + lam)**2))

    # logging.info("Used eigvals")
    # logging.info(eigvals[-r:])

    i11 = (n - m) / (sigma_e**2) + tr_Winv_2 / (sigma_g**2)
    i12 = tr_Winv / (sigma_g**2) - (sigma_e*tr_Winv_2) / (sigma_g**3)
    i22 = m / (sigma_g**2) - (2 * sigma_e * tr_Winv) / (sigma_g**3) + (sigma_e**2 * tr_Winv_2) / (sigma_g**4)

    if constrain_sigma:
        var_mat = 2 / (i22 + i11 - (2*i12))
    else:
        obInfo = 0.5*np.matrix([[i11, i12], [i12, i22]])
        var_mat = np.linalg.inv(obInfo)

    return var_mat

def HEELS_variance_lowrank(sigma_g, sigma_e, LD_b, eigval, eigvec, n, m):
    lam = sigma_e / sigma_g
    W_t = LD_b.copy()
    W_t[0,:] = W_t[0,:] + lam
    inv = chol_fac_banded(W_t, W_t.shape[1])

    Winv = woodbury_matrix(inv, eigvec, eigval)
    Winv_mid = woodbury_matrix(inv, eigvec, 2*eigval)
    Winv_2 = woodbury_matrix(Winv_mid.dot(inv), eigvec, eigval**2)
    tr_Winv = np.sum(np.diag(Winv))
    tr_Winv_2 = np.sum(np.diag(Winv_2))

    i11 = (n - m) / (sigma_e**2) + tr_Winv_2 / (sigma_g**2)
    i12 = tr_Winv / (sigma_g**2) - (sigma_e*tr_Winv_2) / (sigma_g**3)
    i22 = m / (sigma_g**2) - (2 * sigma_e*tr_Winv) / (sigma_g**3) + (sigma_e**2 * tr_Winv_2) / (sigma_g**4)
    obInfo = 0.5*np.matrix([[i11, i12], [i12, i22]])
    var_mat = np.linalg.inv(obInfo)
    return var_mat

def chol_fac_banded(W_t_tilde, m):
    '''Using Cholesky factorization to invert a PSD matrix. '''
    chol = cholesky_banded(W_t_tilde, lower = True)
    precond = cho_solve_banded((chol, True), np.eye(m))

    return precond
    
def woodbury_matrix(inv, eigvec, eigval):
    lowrank = np.diag(1/eigval) + np.transpose(eigvec).dot(inv.dot(eigvec))
    WU = inv.dot(eigvec)
    Abba = WU.dot((np.linalg.inv(lowrank).dot(np.transpose(WU))))
    return inv - Abba

def ML_variance(sigma_g, sigma_e, eigvals):
    logging.info("Computing the variance of ML estimators: ")
    i11 = np.sum(1/(sigma_g*eigvals + sigma_e)**2)
    i12 = np.sum(eigvals / ((sigma_g*eigvals + sigma_e)**2))
    i22 = np.sum((eigvals**2) / ((sigma_g*eigvals + sigma_e)**2))
    obInfo = 0.5*np.matrix([[i11, i12], [i12, i22]])
    var = np.linalg.inv(obInfo)

    return var

