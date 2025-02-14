# ZEDTool 
Z Estimate Diagnostic Tool.

A tool for evaluating and adjusting detections from SMLM experiments.
## Installation

To install ZEDTool from source:

```bash
git clone https://github.com/johnfmarkham/zedtool.git
conda env create --file=environment.yaml
conda activate zedtool-env
cd zedtool
pip install .
```

## Running ZEDTool

Once installed, you can run ZEDTool using:

```bash
python -m zedtool config.yaml
```

where `config.yaml` is a YAML configuration file containing experiment-specific parameters.

## Configuration Options

ZEDTool relies on a YAML file for specifying various options. Below is a detailed explanation of the parameters used in the configuration file:

### **Input and Output**
- `detections_file`: Path to the input file containing detections.
- `output_dir`: Directory where all output files will be stored.
- `binary_detections_file`: Path to a cached binary version of the detections file (used if `make_caches` is enabled).
- `drift_correction_file`: File containing drift correction values (used if `apply_drift_correction` is enabled).

### **Binning and Debugging Options**
- `bin_resolution`: Bin size in nm for the binned image.
- `z_step_step`: Step size in nm for one step in the Z direction (sign must match microscope).
- `debug`: Controls verbosity and plot output (1 for enabled).
- `noclobber`: If set to 1, prevents overwriting cached intermediate results. Safest to leave = 0.
- `make_caches`: If set to 1, saves detections to a binary file for faster subsequent loading.

### **Detection Filtering**
- `n_min_cutoff`: Minimum detection density in the binned x,y image to retain detections.
- `n_max_cutoff`: Maximum detection density in the binned x,y image.
- `select_cols`: Comma-separated list of selection criteria columns. All criteria are *and*-ed together. This can be used for quality filtering and or for processing only a portion of hte field of view.
- `select_ranges`: Comma-separated list of value ranges for filtering detections. Must match select_cols.

### **Frame, z-step, cycle, and time-point Ranges**
- `frame_range`: Range of frames per z-step.
- `z_step_range`: Range of z-steps per cycle.
- `cycle_range`: Number of cycles per time-point.
- `time_point_range`: Number of time-points per experiment.

### **Column Names in Input Data**
These specify how different columns in the detections file are named:
- `frame_col`, `image_id_col`, `z_step_col`, `cycle_col`, `time_point_col`: Identifiers for frames, cycles, and time-points.
- `x_col`, `y_col`, `z_col`: Spatial coordinates.
- `x_sd_col`, `y_sd_col`, `z_sd_col`: Precision values for x, y, and z coordinates.
- `photons_col`: Photon count.
- `chisq_col`: Chi-squared value of the fit (not currently used but may be in future versions).
- `deltaz_col`: Distance of z relative to focal plane or putative biplane midpoint. (default=deltaz)

### **Fiducial Segmentation and Filtering**
- `excluded_fiducials`: Comma-separated list of fiducials to exclude from drift correction.
- `median_filter_disc_radius`: Disc radius for filtering out speckles in the binned image.
- `filling_disc_radius`: Radius for filling holes in thresholded images.
- `dilation_disc_radius`: Dilation radius for catching "wandering" fiducials.
- `min_fiducial_size`: Minimum size (in pixels) of fiducials.
- `min_fiducial_detections`: Minimum number of detections required for a fiducial.
- `max_detections_per_image`: Maximum number of detections allowed per fiducial per image.
- `quantile_tail_cutoff`: Discarded fraction of extreme values for quality control (typically 0.01-0.05). If you have less than
20 fiducials then set this to 0.
- `polynomial_degree`: Polynomial degree for drift correction fitting. Best left at 2.
- `use_weights_in_fit`: Whether to use precision values in the fit.
- `only_fiducials`: If set to 1, assumes all "bright spots" in the image are fiducials. If you want to find the fiducials automatically then set this to 0.

### **Computation and Plotting Options**
You can disable/enable various processing steps and visualizations by setting these values to 0/1. These are listed  below in the order in which they are performed.
- `apply_drift_correction`: Apply pre-made drift correction to detections. If `correct_detections` is set then a drift correction file is written out that can be applied to a set of detections from the same experiment.
- `mask_on_density`: Remove detections based on density in the binned image. Can be useful to remove background that can drag down the threshold for segmentation.
- `plot_detections`: Plot all detections as binned x-y images and 3D projections.
- `make_quality_metrics`: Compute quality metrics for fiducials beforecorrection. Quality metrics are re-computed at the end after all corrections.
- `plot_fiducial_correlations`: Plot correlations between fiducials. Normally not necessary. Can be slow!
- `plot_fiducials`: Plot fiducials before any correction.
- `plot_summary_stats`: Generate summary statistics plots for detections.
- `correct_fiducials`: Correct the z-coordinate of fiducial detections for z-step variations. This changes the x,y,z,... columns and copies them to x_0,y_0,... In addition it writes a file to the output directory called `drift_correction.tsv` that contains these corrections that can be used as described above.
- `correct_detections`: Apply drift correction to all detections.
- `plot_per_fiducial_fitting`: Generate debugging images for individual fiducial drift corrections. Normally not necessary. Can be slow!

## Example Configuration File

Options are read from a yaml configuration file supplied on the command line. Here is an example:

```yaml
# Files and paths
detections_file: /path/to/input.csv 
output_dir: /path/to/output 
binary_detections_file: /path/to/detections.npy 
drift_correction_file: /path/to/drift_correction.tsv 

# Experiment details
frame_range: 0-199 
z_step_range: 0-60 
cycle_range: 0-4 
time_point_range: 0-0 

# Binning and debugging
bin_resolution: 20 
z_step_step: -100 
debug: 1 
noclobber: 1 
make_caches: 1 

# Detection filtering
n_min_cutoff: 1 
n_max_cutoff: 10000000000
select_cols: x,y 
select_ranges: 0-10000,20000-30000 


# Column names specific to the input file format
# The ones below are for Bruker's SRX software
frame_col: frame 
image_id_col: image-ID 
z_step_col: z-step
cycle_col: cycle
time_point_col: time-point
x_col: x
y_col: y
z_col: z
x_sd_col: precisionx
y_sd_col: precisiony
z_sd_col: precisionz
photons_col: photon-count
chisq_col: chisq 
deltazname: deltaz

# Constants used for segmenting, filtering and fitting
excluded_fiducials: 
median_filter_disc_radius: 1 
filling_disc_radius: 10 
dilation_disc_radius: 10 
min_fiducial_size: 100 
min_fiducial_detections: 10 
max_detections_per_image: 1.1 
quantile_tail_cutoff: 0 
polynomial_degree: 2 
use_weights_in_fit: 0 
only_fiducials: 0 

# What steps to perform
apply_drift_correction: 0 
mask_on_density: 0 
make_quality_metrics: 1 
plot_per_fiducial_fitting: 1 
plot_fiducial_correlations: 0 
plot_summary_stats: 0 
plot_detections: 0 
plot_fiducials: 1 
correct_fiducials: 1  
correct_detections: 1 
```


## License

This package is licensed under the BSD 3-Clause License.


