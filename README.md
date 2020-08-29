# Symmetry

## A Python 3 Script for Determining the Symmetry Point Group of Molecules
### Module Requirement

- Python >= 3.7
- Numpy >= 1.17

### About the Script

Still under development... 

The features of this script are:

1. Parse input file and read the coordinates of a molecule. The output data is a 2-D Numpy array, replacing the first column, i.e. the element symbols, to the atomic mass. 

	- for the fomrat of Gaussian 09 input file (e.g. .gjf and .com) in the Cartesian coordinates (implemented)
	- for the .xyz format (implemented)
	- for other formats of input file (need to be developed)
	
2. Calculate the center of mass of the molecule (CoM) and move the coordinates to the CoM.
3. Calculate the geometric center (GC) of the molecule.
4. Calculate the moment of inertia tensor by using the moved coordinates (in step 2).
5. Calculate the principal moment of inertia by diagonalization of the moment of inertia tensor. 
6. Determine the type of molecule based on the relationship between $I_{a}, I_{b}, I_{c}$  
    - if $I_{a} = I_{b} = I_{c}$, then the molecule belongs to a spherical top
    - if $I_{a} = I_{b} < I_{c}$, then the molecule is an oblate symmetric top
    - if $I_{a} < I_{b} = I_{c}$, then the molecule is a prolate symmetric top
    - if $I_{a} < I_{b} < I_{c}$, then the molecule is an asymmetric top
    - if $I_{a} = 0, I_{b} = I_{c}$, then the molecule is a linear molecule
    - if CoM = GC, then the molecule is centrosymmetric 
7. ...
