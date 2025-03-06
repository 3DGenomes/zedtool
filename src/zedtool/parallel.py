#!/usr/bin/env python3
import numpy as np
from typing import Tuple
import pandas as pd
import scipy.ndimage
import scipy.stats
import logging
import multiprocessing

# Parallel version of minimize_fiducial_fit_variance() since scipy.optimize.minimize()
# is not parallelised properly by multiprocessing.Pool()

def optimise_dim(j: int, nfiducials: int, x_ret: np.ndarray, w: np.ndarray) -> np.ndarray:
    dimensions = ['x', 'y', 'z']
    logging.info(f'Grouping fiducials fits to {dimensions[j]}')
    optimise_method = "L-BFGS-B" # L-BFGS-B, Powell, CG, BFGS, Nelder-Mead, TNC, COBYLA, SLSQP, Newton-CG, trust-ncg, trust-krylov, trust-exact, trust-constr
    # TODO: Put in better initial guess, add bounds and Jacobian function
    initial_offsets = np.zeros(nfiducials - 1)
    initial_cost = variance_cost(initial_offsets, x_ret[j,:,:], w[j,:,:])
    result = scipy.optimize.minimize(variance_cost, initial_offsets, args=(x_ret[j,:,:], w[j,:,:]),
                                    method=optimise_method,options = {'maxiter': 1e5, 'disp': True})
    optimal_offsets = result.x
    final_cost = result.fun
    logging.info(f'Initial cost: {initial_cost} Final cost: {final_cost}')
    x_ret[j,:,:] = apply_offsets(optimal_offsets, x_ret[j,:,:])
    return x_ret[j,:,:]

def variance_cost(offsets, x, w):
    x_shifted = apply_offsets(offsets, x)
    # At each time point, calculate the variance of the fiducial fits across time points
    # Weight this by the uncertainties at each time point
    var_t = np.nanvar(x_shifted, axis=0)
    weight_t = np.sum(w, axis=0)
    ret = np.sum(var_t * weight_t)
    # logging.info(f'Cost: {ret} offsets: {offsets}')
    return ret

def apply_offsets(offsets, x):
    nfiducials = x.shape[0]
    x_shifted = x.copy()
    # Apply offset to all fiducials except the first
    x_shifted[1:nfiducials, :] += offsets[:nfiducials-1, np.newaxis]
    return x_shifted

def minimize_fiducial_fit_variance_parallel(x_ft: np.ndarray, xsd_ft: np.ndarray,config: dict) -> Tuple[np.ndarray, np.ndarray]:
    logging.info('minimize_fiducial_fit_variance_parallel')
    # Finds the offsets that minimise the variance of the fiducial fits at each time point
    # One of the options for grouping together fiducials to find a consensus fit
    nfiducials = x_ft.shape[1]
    ndimensions = x_ft.shape[0]
    x_ret = x_ft.copy()
    w = 1 / xsd_ft**2 # weight to be used in the cost function

    with multiprocessing.Pool() as pool:
        results = pool.starmap(optimise_dim, [(j, nfiducials, x_ret, w) for j in range(ndimensions)])

    for j in range(ndimensions):
        x_ret[j,:,:] = results[j]

    return x_ret, xsd_ft


