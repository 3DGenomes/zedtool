# ZEDTool 
Z Estimate Diagnostic Tool.

A tool for evaluating detections from SMLM experiments.
## Installation

To install ZEDTool from source:

```bash
git clone https://github.com/johnfmarkham/zedtool.git
conda env create --file=linux_env.yaml
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
- `bin_resolution`: Bin size in nanometers for the binned image.
- `z_step_step`: Step size in nanometers for one step in the Z direction (sign-sensitive).
- `debug`: Controls verbosity and plot output (1 for enabled).
- `noclobber`: Prevents overwriting cached intermediate results if set to 1.
- `make_caches`: If set to 1, saves detections to a binary file for faster subsequent loading.

### **Detection Filtering**
- `n_min_cutoff`: Minimum detection density in the binned x,y image to retain detections.
- `n_max_cutoff`: Maximum detection density in the binned x,y image.
- `select_cols`: Comma-separated list of selection criteria columns.
- `select_ranges`: Comma-separated list of value ranges for filtering detections.

### **Frame, Z-step, Cycle, and Time-point Ranges**
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

### **Fiducial Segmentation and Filtering**
- `excluded_fiducials`: Comma-separated list of fiducials to exclude.
- `median_filter_disc_radius`: Disc radius for filtering out speckles in the binned image.
- `filling_disc_radius`: Radius for filling holes in thresholded images.
- `dilation_disc_radius`: Dilation radius for catching "wandering" fiducials.
- `min_fiducial_size`: Minimum size (in pixels) of fiducials.
- `min_fiducial_detections`: Minimum number of detections required for a fiducial.
- `max_detections_per_image`: Maximum number of detections allowed per fiducial per image.
- `quantile_tail_cutoff`: Discard fraction of extreme values for quality control (typically 0.01-0.05).
- `polynomial_degree`: Polynomial degree for drift correction fitting.
- `use_weights_in_fit`: Whether to use precision values in the fit.
- `only_fiducials`: If set to 1, assumes all "bright spots" in the image are fiducials.

### **Computation and Plotting Options**
You can enable or disable various processing steps and visualizations:
- `apply_drift_correction`: Apply pre-made drift correction to detections.
- `mask_on_density`: Remove detections based on density in the binned image.
- `make_quality_metrics`: Compute quality metrics for fiducials before and after correction.
- `plot_per_fiducial_fitting`: Generate debugging images for fiducial drift correction.
- `plot_fiducial_correlations`: Plot correlations between fiducials (can be time-consuming).
- `plot_summary_stats`: Generate summary statistics plots for detections.
- `plot_detections`: Plot all detections as binned x-y images and 3D projections.
- `plot_fiducials`: Plot fiducials before correction.
- `correct_fiducials`: Correct the z-coordinate of fiducial detections for z-step variations.
- `correct_detections`: Apply drift correction to all detections based on fiducials.

## Example Usage

Create a configuration file `config.yaml`:

```yaml
detections_file: ./data/input.csv
output_dir: ./results
bin_resolution: 20
z_step_step: -100
debug: 1
noclobber: 1
make_caches: 1
frame_range: 0-199
z_step_range: 0-60
cycle_range: 0-4
time_point_range: 0-0
apply_drift_correction: 1
plot_detections: 1
```

Run the package with:

```bash
python -m zedtool --config config.yaml
```

## License

This package is licensed under the BSD 3-Clause License.


