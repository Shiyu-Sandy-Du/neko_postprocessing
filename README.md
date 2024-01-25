# neko_TSchannel_postpocessing
This repo gives a python script to read field files 'fieldxz0.fld' and then find out the wall-normal maximum of the 2D Fourier amplitudes of streamwise velocity.
Case setup: 2D Tollmien-Schlichting wave in channel flow, Re_c = 5000, amp_perturbation(t=0) = 1e-6
Here in the repo, fields of two time steps are given: t=0, 0.05
The field file already postprocessed by contrib program map_to_equidistant_1d, and thus grids are equidistant in x and z direction (streamwise and spanwise)
Parallel file reading using https://github.com/adperezm/ppymech.git is not supported yet but is intended and encouraged.
