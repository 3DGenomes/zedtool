#!/usr/bin/env python3
import logging

import numpy as np
import scipy
from typing import Tuple
import matplotlib.pyplot as plt
import os


def plot_fourier_correlation(counts_xyz, counts_xy, config) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate and plot the Fourier Shell Correlation (FSC) for the given counts_xyz data.
    Also do Fourier ring correlation (FRC) by binning to 2Ds

    Parameters:
    - counts_xyz: A 3D numpy array with counts in the x, y, and z dimensions.
    - counts_xy: Counts binned into 2D for Fourier ring correlation.
    - x_bins: A numpy array containing the x bin edges.
    - y_bins: A numpy array containing the y bin edges.
    - z_bins: A numpy array containing the z bin edges.
    - config: A dictionary containing configuration parameters for the FSC calculation.

    Returns:
    - fsc: A numpy array containing the FSC values.
    - radii: A numpy array containing the radii corresponding to the FSC values.
    """
    # Placeholder for actual FSC calculation logic
    # This should be replaced with the actual implementation
    binsize = config['bin_resolution']

    # For 2D FRC, we can use the counts_xy data
    spatial_freqs, frc_vals, spatial_resolutions = compute_fsc(counts_xy, binsize, config)
    plot_fsc(spatial_freqs, frc_vals, spatial_resolutions, ndim=2, config=config)
    spatial_freqs, frc_vals, spatial_resolutions = compute_fsc(counts_xyz, binsize, config)
    plot_fsc(spatial_freqs, frc_vals, spatial_resolutions, ndim=3, config=config)

    return spatial_freqs, frc_vals

def plot_fsc(spatial_freqs, frc_vals, spatial_resolutions, ndim, config) -> Tuple[np.ndarray, np.ndarray]:
    """
    Plot the Fourier Shell Correlation (FSC) for the given spatial frequencies and FRC values.

    Parameters:
    - spatial_freqs: A numpy array containing the spatial frequencies.
    - frc_vals: A numpy array containing the FRC values.
    - spatial_resolutions: A numpy array containing the spatial resolutions.
    - ndim : The number of dimensions (2D or 3D) for the FSC calculation.
    Returns:
    - fsc: A numpy array containing the FSC values.
    - radii: A numpy array containing the radii corresponding to the FSC values.
    """
    logging.info(f'Plotting Fourier correlation quality metric: {ndim}d')
    ring_shell = 'Ring' if ndim == 2 else 'Shell'
    plt.figure(figsize=(10, 6))
    plt.plot(spatial_freqs, frc_vals, marker='o', linestyle='-', color='b')
    plt.xlabel('Spatial Frequency (1/nm)')
    plt.ylabel('Correlation')
    plt.title(f"{ndim}D Fourier {ring_shell} Correlation")
    plt.grid()
    outpath = os.path.join(config['output_dir'], f"fourier_cor_vs_f_{ndim}d.{config['plot_format']}")
    plt.savefig(outpath)

    plt.figure(figsize=(6, 4))
    plt.plot(1 / spatial_freqs, frc_vals, label="Correlation")
    plt.axhline(1 / 7, color='r', linestyle='--', label="1/7 threshold")
    # Determine resolution cutoff
    crossings = np.where(frc_vals < 1 / 7)[0]
    if crossings.size > 0:
        cutoff_resolution_nm = spatial_resolutions[crossings[0]]
        logging.info(f"Estimated resolution {ndim}D: {cutoff_resolution_nm:.1f} nm")
        # Write to file
        with open(os.path.join(config['output_dir'], f"fourier_cor_{ndim}d_resolution.txt"), 'w') as f:
            f.write(f"{cutoff_resolution_nm:.1f} nm\n")
        # Write on plot
        plt.text(0.1, 0.3, f"Estimated resolution: {cutoff_resolution_nm:.1f} nm", transform=plt.gca().transAxes,
                    fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
    plt.xlabel("Distance (nm)")
    plt.ylabel("Correlation")
    plt.title(f"{ndim}D Fourier {ring_shell} Correlation")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    outpath = os.path.join(config['output_dir'], f"fourier_cor_vs_x_{ndim}d.{config['plot_format']}")
    plt.savefig(outpath)

    # Save the results to a TSV file with header and cols names spatial_freqs, spatial_resolutions, frc_vals
    outfile = os.path.join(config['output_dir'], f"fourier_cor_{ndim}d.tsv")
    header = 'spatial_freqs\tspatial_resolutions\tfrc_vals\n'
    data = np.column_stack((spatial_freqs, spatial_resolutions, frc_vals))
    np.savetxt(outfile, data, header=header, delimiter='\t', fmt='%.6f')
    return spatial_freqs, frc_vals

def compute_fsc(n, binsize, config) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the Fourier Shell Correlation (FSC) for the given counts_xyz data.

    Parameters:
    - n: A 2D or 3D numpy array with counts in the x, y, and z dimensions.
    - binsize: The size of the bins in nm
    - config: A dictionary containing configuration parameters for the FSC calculation.

    Returns:
    - fsc: A numpy array containing the FSC values.
    - radii: A numpy array containing the radii corresponding to the FSC values.
    """
    logging.info(f'Computing Fourier correlation quality metric: {len(n.shape)}d')
    # Randomly split the counts into two independent images
    n1 = np.random.binomial(n.astype(int), 0.5)
    n2 = n - n1
    # Compute the Fourier transform
    f1 = scipy.fft.fftn(n1)
    f2 = scipy.fft.fftn(n2)
    # Compute shell indices
    shape = n.shape
    # Generate frequency arrays for each axis based on their lengths
    freqs = [np.fft.fftfreq(j, d=binsize) for j in shape]
    # Create a meshgrid of frequencies for all axes
    mesh = np.meshgrid(*freqs, indexing='ij')
    # Compute the radial frequency (Euclidean norm) at each ij')
    kr = np.linalg.norm(np.stack(mesh, axis=-1), axis=-1)

    max_freq = kr.max()
    nbins = 100
    shell_edges = np.linspace(0, max_freq, nbins + 1)
    frc_vals = np.zeros(nbins)
    counts = np.zeros(nbins)

    for i in range(nbins):
        mask = (kr >= shell_edges[i]) & (kr < shell_edges[i + 1])
        num = np.sum(f1[mask] * np.conj(f2[mask])).real
        den = np.sqrt(np.sum(np.abs(f1[mask]) ** 2) * np.sum(np.abs(f2[mask]) ** 2))
        if den > 0:
            frc_vals[i] = num / den
            counts[i] = np.sum(mask)
        else:
            frc_vals[i] = 0

    # Compute corresponding spatial frequencies
    spatial_freqs = 0.5 * (shell_edges[:-1] + shell_edges[1:])
    spatial_resolutions = 1 / spatial_freqs
    return spatial_freqs, frc_vals, spatial_resolutions
