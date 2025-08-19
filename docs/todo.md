### TODO:
* Handle files with missing first col in header like this:
  ```
  ,frame,x,y,z,photons,sx,sy,bg,lpx,lpy
  0,0,170.53981,269.0068,0.5935291,4966.0,0.039154563,0.039881457,21.0,0.15018058,0.15018058
  1,0,177.25436,256.61942,0.699501,7871.0,0.032050293,0.03257835,25.0,0.14770485,0.14770485
  2,0,180.7767,321.80972,0.34599125,5464.0,0.03864466,0.039418157,23.0,0.46158835,0.46158835
  ```
* Find public data on microscope zenobo/ 
* Fail if there are no good fiducials and report why there might be a problem (filtering turned on while using included_fiducial)
* Diagram of workflow
* Multi-pass does not seem to be filtering on goodness of fit
* Fitting can't handle fitting_interval=z_step when there's missing stuff
* Automatically check deltaz_step_step
* Should throw errors for ERROR messages (eg. when there's no fiducials)
* Add option to export only fiducials or good fiducials plus detections 
* Option to exclude fiducials that are missing detections in some time intervals
* In process_detections() reduce unnecessary operations if only appending or plotting segmentation.
* Check that precision is being preserved in corrected_detections,csv (or replaced with something usable - maybe plot old and new)
* For drift_correct_detections_multi_pass, change config['output_dir'] to capture intermediate results.
* Parallelise: - plot_fiducial_correlations(), make_fiducial_stats(),make_drift_corrections(), plot_time_point_metrics()
* Make the quantities used for fiducial selection configurable and plot those quantities in histograms.
* Reinstate max_detections_per_image
* Histogram binned detections gets a log scale on y
* Make a webpage with all relevant output
* Look at Irene's cima code: TB=SG.TransformBlurrer(), TB.SR_gaussian_blur(strOBJ,Precision, 1)
* Contact SMLM authors asking for detection data with fiducials? Or download without from fightclub?
* Add defaults to config file/README.md
* Write tutorial on github
* Select cols to read: pd.read_csv("data.csv", usecols=["D", "B"], dtype={"B": float, "D": int})
  - include columns that SRX needs plus the ones required by zedtool
  - see if data type requiring less data can be specified.
* Optionally add faked error cols if they are not there. Or do without somehow.
* Count detections per fitting range

### DONE:
* df.to_csv () can be slow; use some other alternative?
  - df.to_csv("output.csv", index=False, engine="pyarrow") (require package pyarrow)
  - import polars as pl; pl.from_pandas(df).write_csv("output.csv")
  - df.to_csv("output.csv", index=False, quoting=csv.QUOTE_NONE, chunksize=1000000)
* Try out picasso or other drift correction methods.
* Correlation plots for antonios's bead experiments and find "outliers" beads
* Benchmark an experiment with fiducials at different z's where systematic error in z can fool correction.
* Fiducial images are rotated by 90 degrees in the output. Fix this and add a scale bar.
  - currently +ve x is down and +ve y is right.
* Expand the fiducial image boundary to make space for the label - or whatever's being cut off
* config_base.yaml gets rest of options
* Sometimes fiducials_plot_cropped seems cropped on one side
* Fourier ring quality metric config['plot_fourier_correlation']
* Export time point metrics
* corrected_detections.csv should go into the <output_dir>_corrected directory. Be sure that is made in advance.
* Option to fit with a constant, per sweep
* Test included_fiducials
* Put aside some deltaz results, find that good experiment of guy's
* Look at journals for writeup - what other tools for labelling have been written up (Bintu, SMLM, etc)
* Remove strips without fiducial from non-fiducial image to create a fiducial-only image
* Add (and then remove) is_fiducial column to enable removal of fiducial detections for those that are not good fiducialss
* Read config file from dropbox or other storage
* Check how well dual pass works.
* Correct fiducial for z_step per time_step instead of averaging over all time steps. (Not worth it.)
* binned_detections, detections_image get more sensible names (eg binned_detections_2d and binned_detections_3d)
* Should fiducial_mask and segmentation_mask be png? 
* Save corrected detections in pickle or hd5 as well and allow detections to be read from anything. Not worth it. 
* There sometimes can be nans in corrected fiducials for large experiment with z-step correction. Find out why.
* Say what fiducials are being rejected and why. Should there be outlier rejection on each column or selection of columns?
* Automatically filter based on quality of consensus fit and on time points drift metric
* sd_tail_cutoff, quantile_outlier_cutoff - config file and doco
* drift_correct_detections_multi_pass - config file and doco
* Specify num_threads in config file, add to multiprocessing and doco
* make_time_point_metrics() - extract figure of merit per fiducial
* test new code
* test no cache
* test cpu count
* Make a simpler example config file and a minimum a shorthand 3-pass processing step
* Can a second pass with lower order help? Can multiple passes with different orders help? 
* Check for fiducial areas overlapping and then move overlapping regions of interest
* Speed up optimise_dim() by using a better initial guess, putting in bounds and computing the Jacobian
* Remove unnecessary plots 
* What plots can be removed to speed things up?
* Compute fwhm from histogram fit rather than histogram itself
* Add float_format, llr_col, probe_col, log_likelihood_col doco and config files
* Put filter_fiducials_with_clustering into doco and config files
* Put included_fiducials, plot_time_point_metrics, save_non_fiducial_detections into doco and test
* Put zstep_correct_detections into doco and config files
* Put deconvolution parameters into doco and config file
* Run Guy's experiment with outlier removal pre-fitting
* Run Guy's experiment with plotting fits of fiducials
* Transfer some code from fiducials to drift and to detections
* Rationalise when detections are written - need to be written after filtering for example.
* Drift correct from a drift correction file with just one entry - to allow Peter's stuff to be done.
* Glue together Peter's steps - make a glue tool
* For big experiment, can all detections or fiducials be corrected with deltaz? deltaz_correct_detections()
* included_fiducials to allow for whitelisting of fiducials instead of blacklisting
* Allow n_min_cutoff and n_max_cutoff to be empty
* Does z-step-based fiducial fitting give benefit to fitting either with or without filtering? No clear benefit.
* Plot std dev versus num detections for deconvolution to determine what peaks should be corrected. DONE.
* In deconvolution, do accounting on numbers of peaks and their sizes
* In order to check any spikes in drift, plot d[xyz]dt, averaging over all fiducials/detections, excluding zstep changes
 - doesn't work 
* Look for smlm detections from papers that can be downloaded and processed. At least for getting the format.
* Look into papers on drift correction/QC and see how this could be written up.
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
