#!/usr/bin/env python3

from parse_coord import read_file, symbol_to_mass
from plot_operators import visualization
import argparse as ap
import numpy as np


parser = ap.ArgumentParser(
            add_help=True,
            formatter_class=ap.RawDescriptionHelpFormatter,
            description="""
	The aim of this Python3 script is for determining symmetry point group 
of molecules.

#########################  How to Use this script  ############################

to see help:
python calc_moment_of_inertia.py -h [OR --help]

to run:
python calc_moment_of_inertia.py PATH FILENAME

#########################  Let's try it and enjoy! ############################
""")

parser.add_argument("-p", "--fpath", 
                    action='store_true', help="input file path")
parser.add_argument("-n", "--fname", 
                    action='store_true', help="input file name")

args = parser.parse_args()



periodic_table = {
"H" :[1  , 1.00797],
"He":[2  , 4.0026],
"Li":[3  , 6.939],
"Be":[4  ,],
"B" :[5  ,],
"C" :[6  , 12.01115],
"N" :[7  , 14.0067],
"O" :[8  , 15.9994],
"F" :[9  ,],
"Ne":[10 ,],
"Na":[11 ,],
"Mg":[12 ,],
"Al":[13 ,],
"Si":[14 ,],
"P" :[15 ,],
"S" :[16 ,],
"Cl":[17 ,  34.9689],
"Ar":[18 ,],
"K" :[19 ,],
"Ca":[20 ,],
"Sc":[21 ,],
"Ti":[22 ,],
"V" :[23 ,],
"Cr":[24 ,],
"Mn":[25 ,],
"Fe":[26 ,],
"Co":[27 ,],
"Ni":[28 ,],
"Cu":[29 ,],
"Zn":[30 ,],
"Ga":[31 ,],
"Ge":[32 ,],
"As":[33 ,],
"Se":[34 ,],
"Br":[35 ,],
"Kr":[36 ,],
"Rb":[37 ,],
"Sr":[38 ,],
"Y" :[39 ,],
"Zr":[40 ,],
"Nb":[41 ,],
"Mo":[42 ,],
"Tc":[43 ,],
"Ru":[44 ,],
"Rh":[45 ,],
"Pd":[46 ,],
"Ag":[47 ,],
"Cd":[48 ,],
"In":[49 ,],
"Sn":[50 ,],
"Sb":[51 ,],
"Te":[52 ,],
"I" :[53 ,],
"Xe":[54 ,],
"Cs":[55 ,],
"Ba":[56 ,],
"La":[57 ,],
"Ce":[58 ,],
"Pr":[59 ,],
"Nd":[60 ,],
"Pm":[61 ,],
"Sm":[62 ,],
"Eu":[63 ,],
"Gd":[64 ,],
"Tb":[65 ,],
"Dy":[66 ,],
"Ho":[67 ,],
"Er":[68 ,],
"Tm":[69 ,],
"Yb":[70 ,],
"Lu":[71 ,],
"Hf":[72 ,],
"Ta":[73 ,],
"W" :[74 ,],
"Re":[75 ,],
"Os":[76 ,],
"Ir":[77 ,],
"Pt":[78 ,],
"Au":[79 ,],
"Hg":[80 ,],
"Tl":[81 ,],
"Pb":[82 ,],
"Bi":[83 ,],
"Po":[84 ,],
"At":[85 ,],
"Rn":[86 ,],
"Fr":[87 ,],
"Ra":[88 ,],
"Ac":[89 ,],
"Th":[90 ,],
"Pa":[91 ,],
"U" :[92 ,],
"Np":[93 ,],
"Pu":[94 ,],
"Am":[95 ,],
"Cm":[96 ,],
"Bk":[97 ,],
"Cf":[98 ,],
"Es":[99 ,],
"Fm":[100,],
"Md":[101,],
"No":[102,],
"Lr":[103,],
"Rf":[104,],
"Db":[105,],
"Sg":[106,],
"Bh":[107,],
"Hs":[108,],
"Mt":[109,],
"Ds":[110,],
"Rg":[111,],
"Cn":[112,],
"Nh":[113,],
"Fl":[114,],
"Mc":[115,],
"Lv":[116,],
"Ts":[117,],
"Og":[118,]}

  

def calc_geom_center(data_array):
    """  This function calculates the geometry center of a given data array.
    The format of the input data array is a two dimensional array, which
    contains the x, y, z Cartesion coordinates.
    It returns a 1-D data array.
    """
    return np.average(data_array, axis=0)


def calc_center_of_mass(data_array):
    """  This function calculates the center of mass of a given data array.
    The the input data array is a two dimensional array, containing the 
    x, y, z Cartesion coordinates and their corresponding mass values.
    It returns a 1-D data array.
    """
    return np.average(data_array[:,1:], axis=0, weights=data_array[:,0])


def calc_inertia_tensor(array, mass):
    """  This function calculates the Elements of inertia tensor for the
    moved centered coordinates.
    
    The structure of the array is a two dimensional array, which contains
    the mass of elements (the first column) and their corresponded x, y, z
    Cartesion coordinates.
	
	parameters:
	------------------------------
	array:    The shifted coordinates 
	
    """
    I_xx = (mass * np.sum(np.square(array[:,1:3:1]),axis=1)).sum()
    I_yy = (mass * np.sum(np.square(array[:,0:3:2]),axis=1)).sum()
    I_zz = (mass * np.sum(np.square(array[:,0:2:1]),axis=1)).sum()
    I_xy = (-1 * mass * np.prod(array[:,0:2:1],axis=1)).sum()
    I_yz = (-1 * mass * np.prod(array[:,1:3:1],axis=1)).sum()
    I_xz = (-1 * mass * np.prod(array[:,0:3:2],axis=1)).sum()
    I = np.array([[I_xx, I_xy, I_xz],
		  [I_xy, I_yy, I_yz],
		  [I_xz, I_yz, I_zz]])
    return I


def find_principal_axes(array):
    """  This function finds the principal axes (I_a, I_b, I_c) by using
    diagonalizing the inertia tensor.
	
	parameters:
	------------------------------
	array:    The tensor of the moment of inertia. 
    """
    eig_val, eig_vec = np.linalg.eigh(array)
    D = np.linalg.multi_dot([np.linalg.inv(eig_vec), array, eig_vec])
    D = np.around(D, decimals=6)
    return eig_vec, D, np.diag(D) 


def classify_molecule(I_abc):
    Ia, Ib, Ic = I_abc
    if np.isclose(Ia, Ib):
        if np.isclose(Ib, Ic):
            return "Spherical top"
        return "Oblate symmetric top"
    if np.isclose(Ib, Ic):
        if Ia == 0:
            return "Linear molecule"
        return "Prolate symmetric top"
    return "Asymmetric top"


def main():
    """
    workflow
    """
    filepath = input("Please Enter the Path of the Input file:\n")
    filename = input("Please Enter the File Name (e.g. H2O.gjf):\n")
##    filepath = args.fpath
##    filename = args.fname
    data_array = read_file(filepath, filename, periodic_table)
    print("Input coordinates = \n{:}\n".format(data_array))
    

    data_array = symbol_to_mass(data_array, periodic_table)
    CoM_coord = calc_center_of_mass(data_array)
    print("Center of Mass = \n{:}\n".format(CoM_coord))
    
    new_coord = data_array[:,1:] - CoM_coord
    print("shifted coordinates = \n{:}\n".format(np.column_stack((data_array[:,0],
                                                                  new_coord))))

    CoM_coord = calc_center_of_mass(np.column_stack((data_array[:,0],
                                                     new_coord)))
    print("Center of Mass = \n{:}\n".format(CoM_coord))

    GC_coord = calc_geom_center(new_coord)
    print("Geometry Center = \n{:}\n".format(GC_coord))
    print("overlap or not? {:}\n".format(np.allclose(CoM_coord, GC_coord, rtol=1e-4)))
    
    I = calc_inertia_tensor(new_coord, data_array[:,0])
    eig_vec, D, I_abc = find_principal_axes(I)
    mol_type = classify_molecule(I_abc)
    return CoM_coord, eig_vec, new_coord, I, D, I_abc, mol_type



if __name__ == "__main__":
    CoM_coord, eig_vec, new_coord, I, D, I_abc, mol_type = main()
    print("The inertia_tensor I is \n{:}\n".format(I))
    print("The diagonalized I is \n{:}\n".format(np.diag(D)))
    print("The molecule is {:}".format(mol_type))
    print("I_a = {:}\nI_b = {:}\nI_c = {:}\n".format(*I_abc))
    visualization(CoM_coord, eig_vec, new_coord)
    
