# galaxy_21cm_cross_correlation

## About
This repository compares the two-point function and the k-nearest-neighbour cumulative distribution function formalisms for studying galaxy–21cm cross-correlations during the Epoch of Reionization (EoR).

## Details of the repository
The main script for computing the two-point correlation matrix and generating null samples is  
**cal_2pt_nullcorr_AAstar_FG_subregion_check.py**.  
This script requires **Pylians3** and **pyFFTW**, which are dependencies of the routines in **crosscorrelation_utils.py**.

## Data Files
The directory **reion_files_mixed_compare_cases** contains the true galaxy catalogs and reionization fields.  
These files are used only to determine how many galaxies lie within the chosen subvolume so that the same number of random galaxies can be generated.

The directory **tools21cm_datafiles_with_unq_noise_120hrs_ngrid80** contains **2501 noise realizations** generated using tools21cm package.

## Processing Notes
The wedge is applied to each noise cube, and random galaxies are generated internally within the Python code (see line 381 onward in the main script).


