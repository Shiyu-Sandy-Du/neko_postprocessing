# TSchannel_postpocessing
This folder gives a python script to read field files 'fieldxz0.fld' and then find out the wall-normal maximum of the 2D Fourier amplitudes of streamwise velocity.
Case: Tollmien-Schlichting wave test
The field files should be already postprocessed by contrib program map_to_equidistant_1d, and thus grids are equidistant in x and z direction (streamwise and spanwise)
Parallel file reading using https://github.com/adperezm/ppymech.git is not supported yet but is intended and encouraged.
