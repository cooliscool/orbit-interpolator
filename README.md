# Orbit Interpolator
## Brief

  An Elliptic orbit interpolator which uses (x,y,z,t) data as input and finds Keplerian orbital parameters. 
  
  Using two Algorithms :
  1. Choosing random points (set of 3) the given data set finding all orbital params using Gibb's method. Repeating the same for another random set of points for given number of iterations.
  2. Fitting an Ellipse equation using the given data set in the orbital plane ( averaged ) . (yet to implement)
 
  
  Data model:
  
  Input data should be in the format of 'trackdata.csv' file in project.
  
  Column no. (2,3,4) : (x,y,z)
  
  Column no. (4,5,6) : (x+diff, y+diff, z+ diff) , is the data with Jittery.
  
## Motivation
  For solving problem of 3D orbit determination of a cubesat, project proposed by AerospaceResearch.net , [gsoc17-a03] Lone Pseudoranger:   orbit position data analysis and interpolation (3d).
  
  Based on the data posted by Andreas Hornig,  https://github.com/aerospaceresearch/summerofcode2017/tree/master/gsoc2017/a03-LonePseudorangerOrbitPosition 
  
## Algorithm Explained
Algo 1:

  1. Parse the file and get (x,y,z) set of coordinates.
  2. Choose random points (set of 3) from the data set.
  3. Find orbital parameters.
  4. Repeat above set for given number of iterations.
  5. Output the averaged value of orbital parameters.

Algo 2: (yet to implement)
 1. Parse x, y, z coordinates from file.
 2. Choose random points (set of 3) from the data set and find the direction of orbital plane.
 3. Repeat above set for given number of iterations.
 4. Transform XY plane coordinate system to New Orbital Plane coordinate system.
 5. Fit an elliptical curve equation in the new plane.
 6. Find remaining orbital parameters and output it.
 
## Results
  
### How good is the Code ? 

