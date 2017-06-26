# Q1-Q16 KeplerPORTs
# ***NOT FOR DR25 SEE https://github.com/nasa/KeplerPORTs For official DR25 KeplerPORTs***
Kepler Planet Occurrence Rate Tools

A series of python tools to illustrate use of the Kepler planet occurrence data products and supplement the article 'Terrestrial Planet Occurrence Rates for the Kepler GK Dwarf Sample', Burke et al., 2015, ApJ, submitted.

occurpy.py - Standalone python program to calculate posterior estimates of the planet occurrence rate for any subsample or extrapolated region from Burke et al. (2015).  Depends upon the data table burke_2015_base.txt which contains parameter samples obtained from the MCMC analysis of the planet distribution function model.

test_comp_grid.py - Illustrate use of the KeplerPORTs_utils module to generate a single target detection contour matching Figure 1 of Burke et al. (2015).

KeplerPORTs_utils - Module containing numerous useful functions relevant to occurrence rate calculations.  The highlights are calculating a single target detection contour, kepler_single_comp(), one dimensional detection efficiency curve from Christensen et al., ApJ, submitted, detection_efficiency(), analytic window function from Burke & McCullough (2014), kepler_window_function(), and much more!

KeplerPORTs_readers - Module to read in the Q1-Q16 stellar table of occurrence data products hosted at NExSci and output the results as a dictionary.  Also contains reader for the new Q1-Q17 occurrence data products at NExSci of the full numerical window function and 1-sigma depth function. 
