#!/usr/bin/env python3
import logging
import plotly.express as px

import numpy as np
import matplotlib
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import tifffile
from PIL import Image, ImageDraw, ImageFont

# Prints some debugging plots for an SRX dataset.
# Takes a corrected and an uncorrected table of detections, registers the rows and finds the corrections.
# Write out a table with both corrected and uncorrected z.

matplotlib.use('TkAgg')

def plot_scatter(x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str, title: str, filename: str, config: dict) -> int:
    filetype = "png"
    plt.figure()
    # Scale point size with number of detections
    if len(x) > 0:
        point_size = 100 / len(x)
        point_size = np.max([point_size, 0.01])
        point_size = np.min([point_size, 1.0])
    else:
        point_size = 1.0
        logging.warning(f"Empty x array in plot_scatter: {filename}")
    plt.scatter(x, y, s=point_size)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    # if filename doesn't contain config['output_dir'], prepend it
    if not filename.startswith(config['output_dir']):
        figure_path = os.path.join(config['output_dir'], f"{filename}.{filetype}")
    else:
        figure_path = f"{filename}.{filetype}"
    plt.savefig(figure_path, dpi=300)
    plt.close()
    return 0

def plotly_scatter(x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str, title: str, filename: str, config: dict) -> int:
    filetype = "html"
    figure = px.scatter(x=x, y=y, title=title, labels={xlabel: xlabel, ylabel: ylabel})
    # if filename doesn't contain config['output_dir'], prepend it
    if not filename.startswith(config['output_dir']):
        figure_path = os.path.join(config['output_dir'], f"{filename}.{filetype}")
    else:
        figure_path = f"{filename}.{filetype}"
    figure.write_html(figure_path)
    return 0

def plot_histogram(x: np.ndarray, xlabel: str, ylabel: str, title: str, filename: str, config: dict) -> int:
    hist_bins = 100 # TODO: make this a config option
    filetype = "png"
    plt.figure()
    plt.hist(x, bins=hist_bins)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    # if filename doesn't contain config['output_dir'], prepend it
    if not filename.startswith(config['output_dir']):
        figure_path = os.path.join(config['output_dir'], f"{filename}.{filetype}")
    else:
        figure_path = f"{filename}.{filetype}"
    plt.savefig(figure_path, dpi=300)
    plt.close()
    return 0

def plot_detections_summary(df: np.ndarray, filename: str, config: dict) -> int:
    # plot projections  of det_xyz
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    # scatter plot of x,y in top left
    sc = ax[0, 0].scatter(df[config['x_col']], df[config['y_col']], s=0.01,alpha=0.05, c=df[config['z_step_col']])
    ax[0, 0].set_xlabel('x (nm)')
    ax[0, 0].set_ylabel('y (nm)')
    cbar = plt.colorbar(sc, ax=ax[0, 0])
    cbar.set_label('z-step')
    # scatter plot of x,z in bottom left
    ax[1, 0].scatter(df[config['x_col']], df[config['z_col']], s=0.01,alpha=0.05, c=df[config['z_step_col']])
    ax[1, 0].set_xlabel('x (nm)')
    ax[1, 0].set_ylabel('z (nm)')
    # scatter plot of y,z in top right
    ax[0, 1].scatter(df[config['y_col']], df[config['z_col']], s=0.01,alpha=0.05, c=df[config['z_step_col']])
    ax[0, 1].set_xlabel('y (nm)')
    ax[0, 1].set_ylabel('z (nm)')
    # histogram of z in bottom right
    ax[1, 1].hist(df[config['z_col']], bins=100)
    ax[1, 1].set_xlabel('z (nm)')
    ax[1, 1].set_ylabel('count')
    outfile = os.path.join(config['output_dir'],"detections_summary.png")
    plt.savefig(outfile, dpi=600)
    plt.close()

def plot_binned_detections_summary(n_xy: np.ndarray,mean_xy: np.ndarray, sd_xy: np.ndarray, filename: str, config: dict) -> int:
    fig, ax = plt.subplots(3, 2, figsize=(12, 8))
    plt.subplots_adjust(hspace=0.5)
    im=ax[0, 0].imshow(np.log10(1+n_xy).T,origin='lower')
    ax[0, 0].set_axis_off()
    plt.colorbar(im, ax=ax[0,0], label='log10(n)')
    im=ax[0, 1].imshow(mean_xy.T,origin='lower')
    ax[0, 1].set_axis_off()
    plt.colorbar(im, ax=ax[0,1], label='nm')
    im=ax[1, 0].imshow(sd_xy.T,origin='lower')
    ax[1, 0].set_axis_off()
    plt.colorbar(im, ax=ax[1,0], label='nm')
    ax[1, 1].scatter(np.log10(1 + n_xy).flatten(), sd_xy.flatten(), s=0.1, c=mean_xy.flatten())
    plt.colorbar(im, ax=ax[1, 1], label='mean(z) nm')
    ax[2, 0].scatter(sd_xy.flatten(), mean_xy.flatten(), s=0.1, c=np.log10(1+n_xy).flatten())
    plt.colorbar(im, ax=ax[2, 0], label='log10(n)')
    ax[2, 1].scatter(np.log10(1+n_xy).flatten(), mean_xy.flatten(),s=0.1, c=sd_xy.flatten())
    plt.colorbar(im, ax=ax[2, 1], label='sd(z) nm')
    ax[0, 0].set_title('log_10(n)')
    ax[0, 1].set_title('mean(z)')
    ax[1, 0].set_title('sd(z)')
    ax[1, 1].set_xlabel('log_10(n)')
    ax[1, 1].set_ylabel('sd(z) nm')
    ax[2, 0].set_xlabel('sd(z) nm')
    ax[2, 0].set_ylabel('mean(z) nm')
    ax[2, 1].set_xlabel('log_10(n)')
    ax[2, 1].set_ylabel('mean(z) nm')
    outfile = os.path.join(config['output_dir'],"binned_detections_summary.png")
    fig.savefig(outfile, dpi=600)
    plt.close()

def plot_fiducials(df_fiducials: np.ndarray, df: np.ndarray, config: dict) -> int:
    detections_img_file = 'detections_img.tif'
    fiducials_plot_file = 'fiducials_plot.jpg'
    # read in the image and the segmentation from tifs
    img_filt = tifffile.imread(os.path.join(config['output_dir'], detections_img_file))
    # scale the image to 0-255
    im = 255 * (np.log10(img_filt+1)/np.log10(np.max(img_filt)+1))
    im = im.astype(np.uint8)
    # loop over rows of df
    for j in range(len(df_fiducials)):
        im[int(df_fiducials.min_y[j]):int(df_fiducials.max_y[j]-1), int(df_fiducials.min_x[j])] = 228
        im[int(df_fiducials.min_y[j]):int(df_fiducials.max_y[j]-1), int(df_fiducials.max_x[j]-1)] = 228
        im[int(df_fiducials.min_y[j]), int(df_fiducials.min_x[j]):int(df_fiducials.max_x[j]-1)] = 228
        im[int(df_fiducials.max_y[j]-1), int(df_fiducials.min_x[j]):int(df_fiducials.max_x[j]-1)] = 228
    imp = Image.fromarray(im)
    draw = ImageDraw.Draw(imp)
    for fontname in ["DejaVuSansMono", "couri", "Courier New", "arial", "LiberationSans-Regular"]:
        try:
            font = ImageFont.truetype(fontname, 10)
            break
        except OSError:
            pass
    # Add text
    for j in range(len(df_fiducials)):
        label = df_fiducials.label[j]
        draw.text((df_fiducials.max_x[j], df_fiducials.centroid_y[j]), f"f_{label:04d}", 255,font)
    # imp.show()
    imfile = os.path.join(config['output_dir'], fiducials_plot_file)
    imp.save(imfile, quality=95)

    # foreach roi, plot the dependence of z on config['frame_col'], config['z_step_col'], config['cycle_col']
    for j in range(len(df_fiducials)):
        # columns = [config['frame_col'], config['z_step_col'], 'frame', 'time-point', config['cycle_col']]
        columns = [config['frame_col'], config['z_step_col'], config['cycle_col']]
        for col in columns:
            fiducial_label = df_fiducials.at[j, 'label']
            fiducial_z = np.abs(df_fiducials.at[j,'z_mean']).astype(int)

            outdir = os.path.join(config['fiducial_dir'], f"f_{fiducial_label:04d}_z_{fiducial_z:04d}")
            os.makedirs(outdir, exist_ok=True)
            df_detections_roi = df[df['label'] == fiducial_label]
            if col == config['frame_col']:
                outpath = os.path.join(outdir, f"f_{fiducial_label:04d}_z_{fiducial_z}_z_vs_frame")
                plot_scatter(df_detections_roi[col], df_detections_roi[config['z_col']], 'frame', 'z', "z_vs_frame", outpath, config)
                plotly_scatter(df_detections_roi[col], df_detections_roi[config['z_col']], 'frame', 'z',"z_vs_frame", outpath, config)
            fig, ax = plt.subplots(2, 2, figsize=(12, 9))
            sc = ax[0, 0].scatter(df_detections_roi[config['x_col']], df_detections_roi[config['z_col']], s=0.05, c=df_detections_roi[col], alpha=0.25)
            ax[0, 0].set_xlabel('x')
            ax[0, 0].set_ylabel('z')
            ax[0, 1].scatter(df_detections_roi[config['y_col']], df_detections_roi[config['z_col']], s=0.05, c=df_detections_roi[col], alpha=0.25)
            ax[0, 1].set_xlabel('y')
            ax[0, 1].set_ylabel('z')
            ax[1, 0].scatter(df_detections_roi[config['x_col']], df_detections_roi[config['y_col']], s=0.05, c=df_detections_roi[col], alpha=0.25)
            ax[1, 0].set_xlabel('x')
            ax[1, 0].set_ylabel('y')
            # Use make_axes_locatable to create an inset axis for the colorbar
            divider = make_axes_locatable(ax[1, 1])
            cax = divider.append_axes("right", size="15%", pad=0.2)
            cbar = plt.colorbar(sc, cax=cax, label=col)
            ax[1, 1].set_axis_off()
            outpath = os.path.join(outdir, f"f_{fiducial_label:04d}_z_{fiducial_z}_{col}")
            print(f"Saving {outpath}")
            plt.savefig(outpath, dpi=600)
            plt.close()
    return 0

