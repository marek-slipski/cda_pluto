# Crater Detection Algorithm - Pluto
PHYS 6650 (Optical Remote Sensing) project, "Using edge-detection methods and DEMs to identify and characterize craters on Pluto."
From New Horizons global 300m Digital Elevation Model, slices map into smaller
sections, calculates C-transform and derivatives to detect depressions, and 
determines crater perimeters/centroid/shape coefficients. Location and size-frequency distribution comparison with manually detected
craters (Robbins et al. 2017).

## Prerequistes
Pluto_NewHorizons_Global_DEM_300m_Jul2017_32bit.cub

`pip install -r requirements.txt`

## C-transform
Uses Gaussian-like blur to
focus around lambda-scale neighborhood around the focal pixel along with
gradient of elevation to determine how strongly the slope points toward
the focal pixel. Final C-transform value is an "artificial elevation."

Proposed by Stepinski et al. (2009) - https://doi.org/10.1016/j.icarus.2009.04.026
Used also in Liu et al. (2015) - https://doi.org/10.1007/s11038-015-9467-9

## Results
See reports/Seminar Final Presentation.pdf for more details. C-transform, 
derivative, and shape extraction example, detected depressions, comparison with
Robbins et al. (2017), size frequency distribution.
