import numpy as np
from typing import Tuple
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

    weight_t = np.sum(w[j, :, :], axis=0) / np.sum(w[j, :, :])  # make weight sum to 1 and f(t)
    bounds = make_bounds(x_ret[j, :, :])
    initial_offsets = np.zeros(nfiducials - 1)
    initial_cost = variance_cost(initial_offsets, x_ret[j, :, :], weight_t)
    result = scipy.optimize.minimize(variance_cost, initial_offsets, args=(x_ret[j,:,:], weight_t),
                                    method=optimise_method, jac=True, bounds=bounds, options = {'maxiter': 1e5, 'disp': True})
    optimal_offsets = result.x
    final_cost = result.fun
    logging.info(f'Initial cost: {initial_cost[0]} Final cost: {final_cost}')
    x_ret[j,:,:] = apply_offsets(optimal_offsets, x_ret[j,:,:])
    return x_ret[j,:,:]

def variance_cost(offsets, x, weight_t):
    x_shifted = apply_offsets(offsets, x)
    # At each time point, calculate the variance of the fiducial fits across time points
    # Weight this by the uncertainties at each time point
    var_t = np.nanvar(x_shifted, axis=0)
    ret = np.sum(var_t * weight_t)
    # Make Jacobian
    nfiducials = x.shape[0]
    x_shifted_mean_t = np.nanmean(x_shifted, axis=0)
    jac = (2 / nfiducials) * np.sum((x_shifted - x_shifted_mean_t) * weight_t, axis=1)
    # Remove first element of jac, to match offsets
    jac = jac[1:]
    debug = False
    if debug:
        logging.info(f'Cost: {ret} offsets: {offsets} jac: {jac}')
        logging.info(f'Jacobian: {jac}')
    # combine the cost and jacobian into a tuple
    ret = (ret, jac)
    return ret


def apply_offsets(offsets, x):
    nfiducials = x.shape[0]
    x_shifted = x.copy()
    # Apply offset to all fiducials except the first
    x_shifted[1:nfiducials, :] += offsets[:nfiducials-1, np.newaxis]
    return x_shifted

def make_bounds(x):
    # Make bounds for the offsets
    # Bound is twice the distance from the first fiducial in either direction (overly conservative but better than nothing)
    nfiducials = x.shape[0]
    bounds = []
    for i in range(nfiducials - 1):
        d = np.abs(np.mean(x[i+1,:] - x[0,:]))
        bounds.append((-2*d, 2*d))
    return bounds

def minimize_fiducial_fit_variance_parallel(x_ft: np.ndarray, xsd_ft: np.ndarray,config: dict) -> Tuple[np.ndarray, np.ndarray]:
    logging.info('minimize_fiducial_fit_variance_parallel')
    # Finds the offsets that minimise the variance of the fiducial fits at each time point
    # One of the options for grouping together fiducials to find a consensus fit
    nfiducials = x_ft.shape[1]
    ndimensions = x_ft.shape[0]
    x_ret = x_ft.copy()
    w = 1 / xsd_ft**2 # weight to be used in the cost function

    with multiprocessing.Pool(int(config['num_threads'])) as pool:
        results = pool.starmap(optimise_dim, [(j, nfiducials, x_ret, w) for j in range(ndimensions)])

    for j in range(ndimensions):
        x_ret[j,:,:] = results[j]

    return x_ret, xsd_ft


