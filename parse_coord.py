#!/usr/bin/env python3

from pathlib import Path
import numpy as np



def read_file(filepath, filename, periodic_table):
    """ This function reads Gaussian input file (*.gjf or *.com) and store
    the element symbols and their corresponding coordinates into a list.

    Finally, this function returns a 2d-numpy array with float numbers by
    replacing all element symbols to their atomic masses using thedictionary
    of the periodic table.

    ----------------
    e.g. the coordinates in the input file is
    C     -0.79222718 -0.63527652  0.        
    H     -0.43557275 -1.64408653  0.        
    H     -0.43555434 -0.13087833  0.8736515 
    H     -0.43555434 -0.13087833 -0.8736515 
    H     -1.86222718 -0.63526334  0.        

    the output array is
    [[12.   ,  -0.79222718,  -0.63527652,   0.       ],
    [ 1.0079,  -0.43557275,  -1.64408653,   0.       ],
    [ 1.0079,  -0.43555434,  -0.13087833,   0.8736515],
    [ 1.0079,  -0.43555434,  -0.13087833,  -0.8736515],
    [ 1.0079,  -1.86222718,  -0.63526334,   0.       ]]
    """   
    coord = []
    fullname = Path(filepath, filename)
    print("The directory of input file is:\n{:}\n".format(fullname))
    
    with open(fullname, "r") as fo:
        for line in fo:
            line = line.strip()
            if line.startswith(tuple(periodic_table.keys())):
                coord.append(line.split())
    return symbol_to_mass(coord[1:], periodic_table)


def symbol_to_mass(coord, periodic_table):
    """ This function replace the element symbols i.e. the first
    column of Corrdinates to the corresponding mass. Then return
    the new Corrdinates.
    
    ------------------------
    For instance, the input is
    H2O = [["O",  0.0,  0.0,  0.0],
           ["H",  0.7, -0.7,  0.0],
           ["H", -0.7, -0.7,  0.0]]
    ==>>
    output = [[ 8.    0.    0.0  0.  ]
              [ 1.    0.7  -0.7  0.  ]
              [ 1.   -0.7  -0.7  0.  ]]
    """
    for i in coord:
        if i[0] in periodic_table.keys():
            i[0] = periodic_table[i[0]][1]
    return np.asfarray(coord)


if __name__ == "__main__":
    read_file(filepath, filename, periodic_table)
