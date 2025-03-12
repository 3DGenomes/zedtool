
### TODO:
* Write tutorial on github
* Say what is being rejected and why. Should there be outlier rejection on each column or selection of columns?
* For big experiment, can all detections or fiducials be corrected with deltaz?
* Does z-step-based fiducial fitting give benefit to fitting either with or without filtering?
* Look into papers on drift correction.
* Cache binned detections
* In order to check any spikes in drift, do a d[xyz]dt plot averaging over all fiducials
* Allow n_min_cutoff and n_max_cutoff to be empty
* Speed up optimise_dim() by using a better initial guess, putting in bounds and computing the Jacobian
* Automatically filter based on quality of consensus fit
* Check for fiducial areas overlapping and then move overlapping regions of interest
* There sometimes can be Nans in corrected fiducials for large experiment with z-step correction. Find out why.
* In deconvolution, do accounting on numbers of peaks and their sizes
* Save corrected detections in pickle or hd5 as well and allow detections to be read from anything. 
* Correct fiducial for z_step per time_step instead of averaging over all time steps
* Select cols to read: pd.read_csv("data.csv", usecols=["D", "B"], dtype={"B": float, "D": int})
  - include columns that SRX needs plus the ones required by zedtool
* Parallelise: - plot_fiducial_correlations, make_fiducial_stats, plotting combined fitted corrections
* Put deconvolution parameters into doco and config file
* Put multiprocessing parameter into doco and config files
* Put filter_fiducials_with_clustering into doco and config files
* Add float_format, llr_col, probe_col, log_likelihood_col doco and config files
* Optionally add faked error cols if they are not there
* Optionally compute image-id and zstep if they are not there
* Histogram binned detections gets a log scale on y
* Transfer some code from fiducials to drift and to detections

### DONE:
* Fix sd and zstep plots. Remove cor. Add intercept. Make sd versus diff between co-ord and median.
* Fit model for fiducials/all to deltaz average over fiducials. Try 1-2 steps of big exp
* Plot precision XYZ versus actual distance to see how good an indication it is of actual precision
* Plot fit on plots versus deltaz
* Linear fit to deltaz dependence for fiducials - give it same treatment as correlation and plot it and use it to correct?
* Does quality filtering improve drift correction?
* Check image numbering scheme in the file against parameters
* Check stripes in z for /home/jmarkham/work/zedtool/tests/qf_pass_1/zedtool.log
* For each fiducial, plot z-step correction and z-step-fitting drift correction versus t along with error
* Move parallel code into fiducials, remove duplicates
* Extra bead from Guy
* Put float_format into config file
* Print out less decimal places to tsv files
* Bug in zstep_correct_fiducials_parallel() with k<2? Only z getting corrected. Happens in non-parallel. WHY???
* Dendrograms etc only with debug
* Make fiducials/vx* only on debug.  
* Remove deltaz from some plots, replace with zstep
* Customise and rename covariates plots
* Fix the kmeans2() thing.
* Check that fiducials table and fiducials detections are the same size
* Test that SRX can eat the result
* Better way of finding zstepstep such as max(z) or min(z) over zstep
* Why is 1 fiducial removed from Guy 5 fiucials?
* Test deconvolution on Antonios's data set
* Make example of deconvolution output for larger data set. 
* Improve deconvolution outputs
* Make plt.scatter and plotly.scatter versions proper extensions of the python lib version than cat take all modifications (Not worth it)
* Re-run on big experiment
* How realistic is deconvolution? What proportion of points could be deconvolved - in a time point.
* Fix matplotlib logging by selecting the backend before setting debug level
* Save logfile to output/logfile.txt
* Remove detections.npy and other file options from the config. Put all npy files in cache.
* fwhm returns smoothed version and plot that (removed smoothed version)
* Convert vs wobble quantity to a standard measure by not subtracting average velocity
* Test FWHM on Peter's data
* Make FWHM from histogram fit so that it doesn't get fooled with low numbers of detections
* Put fiducials/hist_*.png into its own directory.
* Move npy files to cache, only write them if cache option selected.
* Add consensus_method to doco
* Make the legend of the fitted fiducials less transparent so that you can see colour properly
* Multiple scatter plots and delta z plots need adaptive spot size.
* fiducials/"*vs_frame" -> vs_t in plot names.
* fiducials/*_fit_vs_frame.* in plot names can go.
* Photon count vs frame gets a rolling average overlaid in a different colour
* "Range for frame not filled:" add min and max to explanation.
* z_mean_per_cycle/frame/z_step (What did I mean by this?)
* correct_fiducials -> zstep_correct_fiducials and correct_detections -> drift_correct_detections
* Remove fiducials on the basis of how well their fits agree with the consensus. 
* Plot histogram of precisionz, deltaz and maybe a surface of precisionz and deltaz
* Save plot tables for z versus zstep to enable zstepstep to be estimated easily
* Need to save fit params, fit quality and consensus quality for each fiducial.
* Add consensus fit quality to fiducial quality table
* Double check the cost function for z-step correction - should it just be the overall weighted variance?
* Change the name of vx, vy, vz to x_madr, y_madr, z_madr 
* Deconvolution of z
* Can main call itself for a second pass to replot fiducials/detections after correction?
* Chase up sd outliers 
* Why do the error bars look suspiciouly uniform on the fiducial fits?
* Delta z or z-step as the thing being corrected? z-step. We don't know deltaz
* Why does deltaz plot go up to 2000 when deltaz is being limited?
* Fix /home/jmarkham/data/2025-02-06_aminoyellow-nanourchins
* Can a flow/diffusion method be used to correct z?
* Document config['binned_detection_file'], save_binned_detections
* Look at some literature on the z-problem
* Stratify fiducials and detections to find x,y,z dependencies?
* Check if x,y has z-step dependence and if so look at correlations.
* Save 2d and 3d tifs for imagej to load
* Print a figure of merit on the z-step correction plots - cost function for before and after z-step correction
* dimnames in config not writeable from config file. Fixed. Change help text to reflect this.
* For each fiducial, plot x,y,z vs deltaz in a 3 panel plot
* Fiducial summary plots - noise and correlation between error and deltaz versus x, y, z. 
* For each fiducial, find correlation of error to deltaz for x,y,z. 
* Do some closer checks on z-step correcting of fiducials
* Run-pass 2 of new version on 3 data sets
* Colour fiducial plots by distance from middle of the focal plane
* Remove cumulative movement from wobbliness score
* Fix colouring of points in debug plots
* Put in config defaults generator
* Put in config checker
* Add plot labels for x,y,z,t
