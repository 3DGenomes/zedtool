# ZEDTool 
Z Estimate Diagnostic Tool.

A tool for evaluating and adjusting detections from SMLM experiments.
## Installation

To install ZEDTool from source:

```bash
git clone https://github.com/3DGenomes/zedtool.git
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

### **Binning and Debugging Options**
- `bin_resolution`: Bin size in nm for the binned image.
- `z_step_step`: Step size in nm for one step in the Z direction. Sign must match microscope setting.
- `debug`: Controls verbosity and plot output. 1=verbose.
- `noclobber`: If set to 1, prevents overwriting cached intermediate results. Safest to leave = 0.
- `make_caches`: If set to 1, saves detections and corrections to binary files for faster subsequent loading.
- `multiprocessing`: If 1 then use multiprocessing to speed up the computation
- `num_threads`: Number of threads to use for multiprocessing. If empty, uses all available threads.
- `float_format`: printf-style format for floating point numbers in outputs files (default %.6g)

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

### **Deconvolution parameters**
- `decon_min_cluster_sd`: Minimum standard deviation of clusters to be considered for deconvolution.
- `decon_sd_shrink_ratio`: Ratio of final/initial standard deviation of clusters following deconvolution.
- `decon_bin_threshold`: Minimum x-y bin threshold to be considered for deconvolution.
- `decon_kmeans_max_k`: Maximum number of clusters in z to be considered for deconvolution.
- `decon_kmeans_proximity_threshold`: Clusters closer than (SD1+SD2)*proximity_threshold are merged.
- `decon_kmeans_min_cluster_detections`: Minimum number of detections in a cluster to be considered for deconvolution.

### **Column Names in Input Data**
These specify how different columns in the detections file are named:
- `frame_col`, `image_id_col`, `z_step_col`, `cycle_col`, `time_point_col`: Identifiers for frames, cycles, and time-points.
- `x_col`, `y_col`, `z_col`: Spatial coordinates.
- `x_sd_col`, `y_sd_col`, `z_sd_col`: Precision values for x, y, and z coordinates.
- `photons_col`: Photon count.
- `chisq_col`: Chi-squared value of the fit (not currently used but may be in future versions).
- `deltaz_col`: Distance of z relative to focal plane or putative biplane midpoint. (default=deltaz)
- `log_likelihood_col`: log likelihood of the fit - currently not used
- `llr_col`: llr log likelihood ratio of the fit - currently not used
- `probe_col`: probe ID when more than one channel is present

### **Fiducial Segmentation and Filtering**
- `excluded_fiducials`: Comma-separated list of fiducials to exclude from drift correction.
- `inluded_fiducials`: Comma-separated list of fiducials to use for drift correction (use _only_ these).
- `median_filter_disc_radius`: Disc radius for filtering out speckles in the binned image.
- `filling_disc_radius`: Radius for filling holes in thresholded images.
- `dilation_disc_radius`: Dilation radius for catching "wandering" fiducials.
- `min_fiducial_size`: Minimum size (in pixels) of fiducials.
- `min_fiducial_detections`: Minimum number of detections required for a fiducial.
- `max_detections_per_image`: Maximum number of detections allowed per fiducial per image.
- `quantile_outlier_cutoff`: Discarded fraction of extreme values for quality control (typically 0.01-0.05). If you have less than
20 fiducials then set this to 0.
- `sd_outlier_cutoff`: Standard deviation cutoff for outlier detection (typically around 2). If you have less than 20 fiducials then set this to 0. 
If non-zero then quantile_outlier_cutoff is ignored. 
- `filter_fiducials_with_clustering`: If 1, then use pre-drift correction clustering to filter fiducials. If you have less than
100 fiducials then set this to 0.
- `polynomial_degree`: Polynomial degree for drift correction fitting. Best left at 2.
- `use_weights_in_fit`: Whether to use precision values in the fit.
- `only_fiducials`: If set to 1, assumes all "bright spots" in the image are fiducials. If you want to find the fiducials automatically then set this to 0.
- `consensus_method`: Method for determining the consensus z value for a fiducial. Options are 'weighted_mean' and 'median' (default).

### **Appending another experiment**
Done before all other operations when `concatenate_detections` is set.
- `concatenate_detections_file`: /your/path/to/experiment.csv # file to concatenate with the current experiment
- `concatenate_offset_file`: /your/path/to/offsets.csv # offsets to apply to the detections in the concatenate_detections_file
The file `concatenate_offset_file` contains one row with the offsets to add.
Aside from `image_id_col`, `time_point_col` and `x_col`, `y_col`, `z_col`, columns all values should be zero.
The columns `image_id_col` and `time_point_col` are used to make the corresponding columns in the output file consistent.
The columns  `x_col`, `y_col` and `z_col` are the spatial offsets _added_ to the detections in the `concatenate_detections_file`.

### **Applying a pre-computed drift correction**
- `drift_correction_file`: File containing drift correction values (used if `apply_drift_correction` is enabled).
If `apply_drift_correction` is set then the drift corrections in `drift_correction_file` are applied. 
This is done directly after any concatenation of other detections but before any other processing.
The values in the column `image_id_col` in `drift_correction_file` must match.
 
### **Computation and Plotting Options**
You can disable/enable various processing steps and visualizations by setting these values to 0/1. These are listed  below in the order in which they are performed.
- `concatenate_detections`: Concatenate all detections from concatenate_detections_file using offsets from concatenate_offset_file.
- `apply_drift_correction`: Apply pre-made drift corrections from `drift_correction_file` to detections. If `correct_detections` is set then a drift correction file is written out that can be applied to a set of detections from the same experiment.
- `plot_detections`: Plot all detections as binned x-y images and 3D projections.
- `mask_on_density`: Remove detections based on density in the binned image. Can be useful to remove background that can drag down the threshold for segmentation.
- `make_quality_metrics`: Compute quality metrics for fiducials beforecorrection. Quality metrics are re-computed at the end after all corrections.
- `plot_fiducial_correlations`: Plot correlations between fiducials. Normally not necessary. Can be slow!
- `plot_fiducials`: Plot fiducials before any correction.
- `plot_summary_stats`: Generate summary statistics plots for detections.
- `plot_time_point_metrics`: Plot time-point metrics of drift correction for fiducials.
- `plot_per_fiducial_fitting`: Make debugging images showing extraction of drift correction from fiducials at each time step
- `zstep_correct_fiducials`: Correct the z-coordinate of fiducial detections for z-step variations. Experimental. 
- `drift_correct_detections`: Do fiducial-based drift correction on all detections. This changes the x,y,z,... columns and copies them to x_0,y_0,... In addition it writes a file to the output directory called `drift_correction.tsv` that contains these corrections that can be used as described above.
- `drift_correct_detections_multi_pass`: Do fiducial-based drift correction on all detections using multiple pass method.
- `deltaz_correct_detections`: Correct z co-ordinate of all detections for deltaz variation
- `deconvolve_z`: Reduce variation in z using a deconvolution-like approach. Experimental. 
- `save_non_fiducial_detections` Save non-fiducial detections to a separate file, `non_fiducial_detections.tsv`
## Example Configuration File
Options are read from a yaml configuration file supplied on the command line. Here is an example:

```yaml
# Files and paths
detections_file: /path/to/input.csv 
output_dir: /path/to/output 

# Experiment details
frame_range: 0-199 
z_step_range: 0-60 
cycle_range: 0-4 
time_point_range: 0-0 

# Binning and debugging
bin_resolution: 20 
z_step_step: -100 

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
deltaz_col: deltaz

# Constants used for segmenting, filtering and fitting
excluded_fiducials: 
median_filter_disc_radius: 1 
filling_disc_radius: 10 
dilation_disc_radius: 10 
min_fiducial_size: 100 
min_fiducial_detections: 10 
max_detections_per_image: 1.1 
polynomial_degree: 2 
only_fiducials: 0 

# What steps to perform
make_quality_metrics: 1 
plot_summary_stats: 1
plot_detections: 1 
plot_fiducials: 1 
drift_correct_detections: 1
plot_time_point_metrics: 1
make_quality_metrics: 1 
```


## License

This package is licensed under the BSD 3-Clause License.


