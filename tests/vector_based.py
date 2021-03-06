#!/usr/bin/env python3

#from parse_coord import read_file, symbol_to_mass
import argparse as ap
import itertools as it
import numpy as np
from scipy.spatial.transform import Rotation


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
python calc_moment_of_inertia.py PATH FILENAME TOLERANCE

#########################  Let's try it and enjoy! ############################
""")

parser.add_argument("-p", "--fpath", 
                    action='store_true', help="input file path")
parser.add_argument("-n", "--fname", 
                    action='store_true', help="input file name")
parser.add_argument("-t", "--tolerance", 
                    action='store_true', help="input tolerance")

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


def calc_geom_center(array):
    """  This function calculates the geometry center of a given data array.
    The format of the input data array is a two dimensional array, which
    contains the x, y, z Cartesion coordinates.

    parameters:
    ---------------------------
    array:  the shifted coordinates (2D-array) of the input molecule.

    It returns the coordinates of the geometry center (1-D array).
    """
    return np.average(array, axis=0)


def calc_center_of_mass(array):
    """  This function calculates the center of mass of a given data array.
    The the input data array is a two dimensional array, containing the 
    x, y, z Cartesion coordinates and their corresponding mass values.
    
    parameters:
    ---------------------------
    array:    The (2D-array) contains atomic mass/number (the 1st column)
            and the Cartesian coordinates of the input molecule.

    It returns the coordinates of the center of mass (1-D array).
    """
    return np.average(array[:,1:], axis=0, weights=array[:,0])


def calc_distance(array1, array2):
    """ This function calculates the distances between two points.
    
    parameters:
    ---------------------------
    array1:  the coordinates (2D-array) of molecule.
    array2:  the coordinates (2D-array) of molecule.

    It returns the distance between two input array (elements)
    """
    diff = array1[:,np.newaxis,:] - array2[np.newaxis,:,:]
    return np.linalg.norm(diff, axis=-1)


def check_inversion(data_array, atom_num):
    """ This function checks whether the molecule contains an inversion center.
    
    parameters:
    ---------------------------
    data_array:  the shifted coordinates (2D-array) of molecule.
    atomic_num:  the atomic number represents the elements (1D-array)
                of the input structure.
    
    It returns the "True" or "False". If "True", the coordinates
    of the inversion center (i) also returns.
    """
    original = np.column_stack((atom_num, data_array))
    inversed = np.column_stack((atom_num, -data_array))
    
    if np.allclose(np.sort(original, axis=0), np.sort(inversed, axis=0)):
        return True#, "Inversion"
    else:
        return False#, "No inversion"


def check_linear(data_array):
    """ This function checks whether the molecule is linear or not.
    
    parameters:
    ---------------------------
    data_array:  the shifted coordinates (2D-array) of molecule.

    It returns the "True" or "False".
    "True" stands for the linear molecule.
    "False" means the non-linear molecule.
    """
    normal_vec = np.cross(data_array[:,np.newaxis,:],
                          data_array[np.newaxis,:,:])
    chk = np.zeros(normal_vec.shape)
    if np.allclose(normal_vec, chk, rtol=1e-6):
        return True#, "Linear"
    else:
        return False#, "Non-linear"


def check_planar(data_array, norm_vecs, CoM_coord):
    """ This function checks whether the molecule is planar or not.
    
    parameters:
    ---------------------------
    data_array:  the shifted coordinates (2D-array) of molecule.
    norm_vecs:   the nornal vectors of the plane spanning by every two position
               vectors (elements).
    CoM_coord:   the center of mass of the molecule.
    

    It returns the "True" or "False".
    "True" stands for the planar molecule.
    "False" means the non-planar molecule.
    """
    sub = data_array - CoM_coord
    dot_product = np.dot(norm_vecs[0], sub.T)
    chk = np.zeros(dot_product.shape)
##    print("The dot product is \n{:}\n".format(dot_product))
##    print("check zeros array \n{:}\n".format(chk))
    if np.allclose(dot_product, chk, rtol=1e-6):
        return True#, "Planar"
    else:
        return False#, "Non-planar"


def find_circle_axis(array):
    """ This function finds the circle axis of the molecule.
    
    P_k^{\prime} = P_k + sin(\phi) (d \times P_k) + (1-cos(\phi)) (d \times (d \times P_k))
    
    (d \times P_k) rotates the P_k 90 degree counterclockwise,
    (d \times (d \times P_k)) rotates the upper result another 90 degree counterclockwise, 
    
    parameters:
    ---------------------------
    array:        the shifted coordinates (2D-array) of molecule.
    norm_vecs:    the nornal vectors of the plane spanning by every two position
                vectors (elements).
    """
    for ind, pair in enumerate(it.combinations(range(array.shape[0]),2)):
        norm_vec = np.croos(array[pair[0]], array[pair[1]])
        for x in range(6, 1, -1):
            pass
    #d_vec =
    #n_vec = 
    sin_phi = np.sin(2*np.pi/x)
    cos_phi = np.cos(2*np.pi/x)
    return x


def search_rotation_axis(array):
    """
    """
    return pass



def find_rotation_symm(ori_array, axis, n_fold):
    """ This function checks whether the n-fold rotational symmetry exists by
    means of the Rotation method from the scipy.spatial.transform module.

    parameters:
    ------------------------------------
    ori_array:  The input data array (2D).
    axis:       The rotational axis.
    n_fold:     The n^th-fold of rotation, for 2 * np.pi / n_fold.
    
    Example array for test:
    ori_array = np.array([
    [ 0.00000000,   1.39499067,   0.00000000],
    [-1.20809736,   0.69749534,   0.00000000],
    [-1.20809736,  -0.69749534,   0.00000000],
    [ 0.00000000,  -1.39499067,   0.00000000],
    [ 1.20809736,  -0.69749533,   0.00000000],
    [ 1.20809736,   0.69749534,   0.00000000]])

    rotation axis for test:
    axis = np.array([0,0,1])
    
    """
    axis = axis / np.linalg.norm(axis)

    theta = 2 * np.pi / n_fold
    rot = Rotation.from_rotvec(theta * axis)
    rot_array = rot.apply(ori_array)
    print("The original vectors \n{:}\n".format(v))
    print("The rotated vectors \n{:}\n".format(new_v))
    
    if np.allclose(np.sort(ori_array, axis=0),
                   np.sort(rot_array, axis=0), rtol=1.e-6):
        return "True"#, n_fold
    else:
        return "False"#, n_fold




def reduce_dimension(array):
    """ This function reduce the dimension of input array.

    parameters:
    ---------------------------
    array:    The 3d-coordinates of molecule.

    The 3d-coordinates was generated by new_coord[:,np.newaxis,:] and
    new_coord[np.newaxis,:,:]. For instance,

    array([[[ 0.        ,  0.        , -0.06199414],    <-- (0,0)
            [ 0.        , -0.39198795,  0.21503555],    <-- (0,1)
            [ 0.        ,  0.39198795,  0.21503555]],   <-- (0,2)

           [[ 0.        , -0.39198795,  0.21503555],    <-- (1,0)
            [ 0.        , -0.78397589,  0.49206524],    <-- (1,1)
            [ 0.        ,  0.        ,  0.49206524]],   <-- (1,2)

           [[ 0.        ,  0.39198795,  0.21503555],    <-- (2,0)
            [ 0.        ,  0.        ,  0.49206524],    <-- (2,1)
            [-0.        ,  0.78397589,  0.49206524]]])  <-- (2,2)
    where (0,0), (1,1), (2,2) are the results from the same atoms, which should
    be ignored.
    while (0,1) and (1,0); (0,2) and (2,0); (1,2) and (2,1) are redundant results.
    
    In this function, the it.combinations(range(dim),2) method was implemented
    to solve to problem.

    It returns a array with the non-repeated calculations.
    """
    dim = array.shape[0]
    new_array = np.zeros((dim, dim))
    for ind, ind_pair in enumerate(it.combinations(range(dim),2)):
        new_array[ind] = array[ind_pair]
    return new_array


def find_reflective_plane(pos1, pos2, CoM, mol_type):
    """ This function finds the reflective plane of the molecule.

    P0 = (P_i + P_j)/2
    n0 = (P_j - P0) / |P_j - P0|  (with the (a,b,c) components)
    
    for theReflective  P'_k,
        x'_k = x_k + 2 * factor * a
        y'_k = y_k + 2 * factor * b
        z'_k = z_k + 2 * factor * c
        where, factor = n0 \cdot (P_0 - Pk) / |n0|
        
    parameters:
    ---------------------------
    pos1:     the element coordinates (x,y,z) of molecule.
    pos2:     the element coordinates (x,y,z) of molecule.
    CoM:      the coordinates of the center of mass.
    mol_type: the type of molecule, for checking if plannar.
    """
    if mol_type == "planar":
        print("+++++++++++{}++++++++++++".format(mol_type))
        nv = pos1 - (pos1 + pos2)/2
        nv /= np.linalg.norm(nv)
        ref = np.zeros((2,3))
        for i, p in enumerate((pos1, pos2)):
            factor = np.dot(nv, (CoM - p))
            ref[i] = p + 2 * factor * nv
            print("====\n{:}\n{:}\n====".format(ref[i], p))
        if np.allclose(ref[0], pos2, rtol=1e-6) and np.allclose(ref[1], pos1, rtol=1e-6):
            print("++++++++++++++++++++ reflection mirror found ++++++++++++++++++++")
            return True#, "$\sigma_v$"
        else:
            print("----\n{:} and {:} does NOT reflective\n----".format(pos1, pos2))
            return False#, "only a planar molecule"
    elif mol_type == "linear":
        pass
    else:
        pass
    
    #### the 1st normal vector perpendicular to plane
##    p0 = (array[:,np.newaxis,:] + array[np.newaxis,:,:]) / 2
##    dim = array.shape[0]
##    new_P0 = np.zeros((dim, dim))
##    
##    for ind, ind_pair in enumerate(it.combinations(range(dim),2)):
##        new_P0[ind] = p0[ind_pair]
##    
##    n0 = array[:,np.newaxis,:] - new_P0[np.newaxis,:,:]    
##    new_n0 = np.zeros((dim, dim))
##    for ind, ind_pair in enumerate(it.combinations(range(dim),2)):
##        new_n0[ind] = n0[ind_pair]
##    new_n0 /= new_n0 / np.linalg.norm(new_n0, axis=-1)
##    
##    
##    #### the 2nd normal vector perpendicular to plane
##    new_P1 = np.zeros((dim, dim))
##    n1 = np.cross(array[:,np.newaxis,:], array[np.newaxis,:,:])
##    for ind, ind_pair in enumerate(it.combinations(range(dim),2)):
##        new_n1[ind] = n1[ind_pair]
##
##    
##    #### the 3rd normal vector perpendicular to plane
##    n2 = np.cross(new_n0[:,np.newaxis,:], new_n1[np.newaxis,:,:])
##    new_n2 = np.zeros((dim, dim))
##    for ind, ind_pair in enumerate(it.combinations(range(dim),2)):
##        new_n2[ind] = n2[ind_pair]
##
####    factor = np.dot(n0, p0[i] - data_array) / np.linalg.norm(n0)
####    reflection = data_array + 2 * factor * n0
####    if np.allclose(reflection, data_array, rtol=1e-6):
####        print("\sigma_v")






def visualization(CoM_coord, rot_vec, new_coord):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    vlen = np.linalg.norm(eig_vec)
    X, Y, Z = CoM_coord
    U, V, W = rot_vec
    ax.quiver(X, Y, Z, U, V, W, pivot='tail',
              length=vlen, arrow_length_ratio=0.2/vlen)

    ax.scatter(new_coord[:,0], new_coord[:,1], new_coord[:,2],
               color="r", marker="o", s=50)
    
    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([0.75, 0.75, 1, 1]))
    ax.set_xlim([new_coord.min(),new_coord.max()])
    ax.set_ylim([new_coord.min(),new_coord.max()])
    ax.set_zlim([new_coord.min(),new_coord.max()])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

    
def main(data_array):
    """
    workflow
    """
    #filepath = input("Please Enter the Path of the Input file:\n")
    #filename = input("Please Enter the File Name (e.g. H2O.gjf):\n")
##    filepath = args.fpath
##    filename = args.fname
##    tolerance = args.tolerance
##    data_array = read_file(filepath, filename, periodic_table)
    print("Input coordinates = \n{:}\n".format(data_array))
    

    data_array = symbol_to_mass(data_array, periodic_table)
    CoM_coord  = calc_center_of_mass(data_array)
    print("Center of Mass = \n{:}\n".format(CoM_coord))
    
    new_coord = data_array[:,1:] - CoM_coord
    print("shifted coordinates = \n{:}\n".format(np.column_stack((data_array[:,0],
                                                                  new_coord))))

    CoM_coord  = calc_center_of_mass(np.column_stack((data_array[:,0],
                                                      new_coord)))
    print("Center of Mass = \n{:}\n".format(CoM_coord))

    GC_coord = calc_geom_center(new_coord)
    print("Geometry Center = \n{:}\n".format(GC_coord))
    print("overlap or not? {:}\n".format(np.allclose(CoM_coord, GC_coord, rtol=1e-4)))
    
    normal_vectors = np.cross(new_coord[:,np.newaxis,:],
                              new_coord[np.newaxis,:,:])
    print("The normal vectors are \n{:}\n".format(normal_vectors[0]))
    
    if check_linear(new_coord):
        if check_inversion(new_coord, data_array[:,0]):
            return "$D_{\infty h}$"#, np.nan
        return "$C_{\infty v}$"#, np.nan
    
    if check_planar(new_coord, normal_vectors, CoM_coord):
        if check_inversion(new_coord, data_array[:,0]):
            count = 0
            for ind, pair in enumerate(it.combinations(range(new_coord.shape[0]),2)):
                print("# {:} # the input points are:\n{:}\n{:}".format(ind, new_coord[pair[0]], new_coord[pair[1]]))
                if find_reflective_plane(new_coord[pair[0]], new_coord[pair[1]], CoM_coord, "planar"):
                    count += 1
                    return "planar and planar has \sigma_h, \sigma_v and inversion center "
            return "planar and planar has \sigma_h and inversion center"
        else:
            count = 0
            for ind, pair in enumerate(it.combinations(range(new_coord.shape[0]),2)):
                print("# {:} # the input points are:\n{:}\n{:}".format(ind, new_coord[pair[0]], new_coord[pair[1]]))
                if find_reflective_plane(new_coord[pair[0]], new_coord[pair[1]], CoM_coord, "planar"):
                    count += 1
                    return "planar and planar has \sigma_h and \sigma_v"#, count
        return "planar"#, np.nan
    
    axis = search_rotation_axis(array):
    if find_rotation_symm(ori_array, axis, n_fold)
        return pass



## =================  pseudocode for determining point group =================
##def determine_point_group(array):
##    if check_linear(array):
##        if check_inversion_center(array):
##            return "$ D_{\infity h} $"
##        return "$ C_{\infity v} $"
##    
##    if check_rotation(array):
##        if count_axis >= 2:
##            if count_C5:
##                if check_inversion_center(array):
##                    return "$ I_h $"
##                else:
##                    return "$ I $"
##            else:    ## NO C5 axis
##                if check_inversion_center(array):
##                    return "$ O_h $"
##                else:
##                   return "$ O $"
##            
##        elif count_axis == 1:    ## ONLY One principal axis
##            if count_C2:
##                if check_planar(array):    ## a horizotal plane
##                    return "$ D_{:}h $".format(n)
##                elif :    ## a vertial plane ########### to do
##                    return "$ D_{:}d $".format(n)
##                else:
##                    return "$ D_{:}$ ".format(n)
##            elif count_C3:
##                if check_planar(array):    ## a horizotal plane
##                    return "$ T_{:}h $"
##                elif :    ## a vertial plane $ \sigma_d $  ########## to do
##                    return "$ T_{:}d $"
##                else:
##                    return "$ T_{:} $"
##                
##            else:    ### NO C2 axis perpendicular to th principal axis
##                if check_planar(array):    ## a horizotal plane
##                    return "$ C_{:}h $".format(n)
##                elif :    ## a vertial plane $ \sigma_d $  ########## to do
##                    return "$ C_{:}v $".format(n)
##                elif check_inversion_center(array):    ## S_{2n}
##                    return "$ S_2{:} $".format(n)
##                else:
##                    return "$ C_{:} $".format(n)
##                
##        else:    ## count_axis == 0, NO rotational axis
##            if check_planar(array):
##                return "$ C_s $"
##            elif check_inversion_center(array):
##                return "$ C_i $"
##            else:
##                return "$ C_1 $"
## ## =========================================================================   



###############################################################################
###########################    examples for test    ###########################
## H2O
##a = [
##["O",    0.00000000,    0.00000000,   -0.11081188],
##["H",    0.00000000,   -0.78397589,    0.44324750],
##["H",   -0.00000000,    0.78397589,    0.44324750]]


#### H2O2
##a = [
##["H",   -1.11122905,    0.58364616,    0.40219369],
##["O",   -0.64803507,   -0.12510213,   -0.05027421],
##["O",    0.64803507,    0.12510213,   -0.05027421],
##["H",    1.11122905,   -0.58364616,    0.40219369]]


#### NH3
##a = [ 
##["N",  0.    ,   0.    ,   0.    ],
##["H",  0.    ,  -0.9377,  -0.3816],
##["H",  0.8121,   0.4689,  -0.3816],
##["H", -0.8121,   0.4689,  -0.3816]]


#### CH3Cl
##a = [
##["C" ,  0.    ,   0.    ,   0.    ], 
##["Cl",  0.    ,   0.    ,   1.7810],
##["H" ,  1.0424,   0.    ,  -0.3901],
##["H" , -0.5212,   0.9027,  -0.3901],
##["H" , -0.5212,  -0.9027,  -0.3901]]


#### HCl
##a = [
##["Cl",     0.80717486,   -0.71001494,    0.00000000],
##["H" ,    -0.48282514,   -0.71001494,    0.00000000]]


#### HCN
##a = [
##["C",   0.00000000,   0.00000000,   -0.49687143],
##["H",   0.00000000,   0.00000000,   -1.56687143],
##["N",   0.00000000,   0.00000000,    0.64972857]]


#### CO2
##a = [
##["C",     -0.39214687,   -0.78171310,    0.01418586],
##["O",     -1.65054687,   -0.78171310,    0.01418586],
##["O",      0.86625313,   -0.78171310,    0.01418586]]


#### C2H2
##a = [
##["C",   -0.00000000,   -0.00000000,   -0.60060000],
##["H",   -0.00000000,   -0.00000000,   -1.67060000],
##["C",    0.00000000,   -0.00000000,    0.60060000],
##["H",    0.00000000,   -0.00000000,    1.67060000]]


#### CH4
##a = [
##["C",   0.    ,   0.    ,   0.    ],
##["H",   0.6276,   0.6276,   0.6276],
##["H",   0.6276,  -0.6276,  -0.6276],
##["H",  -0.6276,   0.6276,  -0.6276],
##["H",  -0.6276,  -0.6276,   0.6276]]


#### C2H4
##a = [
##["C",    0.00000000,   -0.67759997,   0.00000000],
##["H",    0.92414474,   -1.21655197,   0.00000000],
##["H",   -0.92414474,   -1.21655197,   0.00000000],
##["C",    0.00000000,    0.67759997,   0.00000000],
##["H",   -0.92414474,    1.21655197,   0.00000000],
##["H",    0.92414474,    1.21655197,   0.00000000]]


#### C6H6
##a = [
##["C",    0.00000000,   1.39499067,   0.00000000],
##["C",   -1.20809736,   0.69749534,   0.00000000],
##["C",   -1.20809736,  -0.69749534,   0.00000000],
##["C",    0.00000000,  -1.39499067,   0.00000000],
##["C",    1.20809736,  -0.69749533,   0.00000000],
##["C",    1.20809736,   0.69749534,   0.00000000],
##["H",    0.00000000,   2.49460097,   0.00000000],
##["H",   -2.16038781,   1.24730049,   0.00000000],
##["H",   -2.16038781,  -1.24730049,   0.00000000],
##["H",    0.00000000,  -2.49460097,   0.00000000],
##["H",    2.16038781,  -1.24730048,   0.00000000],
##["H",    2.16038781,   1.24730049,   0.00000000]]


#### coronene
##a = [
##["C",    0.68444198,   -3.67907288,    0.00000000],
##["C",    1.41889714,   -2.45760193,   -0.00000000],
##["C",    0.71456145,   -1.23765674,    0.00000000],
##["C",   -0.71456145,   -1.23765674,    0.00000000],
##["C",   -1.41889714,   -2.45760193,   -0.00000000],
##["C",   -0.68444198,   -3.67907288,    0.00000000],
##["C",    1.42912290,   -0.00000000,    0.00000000],
##["C",   -1.42912290,   -0.00000000,    0.00000000],
##["C",   -0.71456145,    1.23765674,    0.00000000],
##["C",    0.71456145,    1.23765674,    0.00000000],
##["C",   -1.41889714,    2.45760193,   -0.00000000],
##["C",   -2.84394959,    2.43228059,   -0.00000000],
##["C",   -3.52839157,    1.24679229,    0.00000000],
##["C",   -2.83779428,   -0.00000000,   -0.00000000],
##["C",   -3.52839157,   -1.24679229,   -0.00000000],
##["C",   -2.84394959,   -2.43228059,    0.00000000],
##["H",   -3.38312959,   -3.39195178,    0.00000000],
##["H",   -4.62908120,   -1.23390028,   -0.00000000],
##["H",    1.24595161,   -4.62585206,    0.00000000],
##["H",   -1.24595161,   -4.62585206,    0.00000000],
##["H",   -3.38312959,    3.39195178,   -0.00000000],
##["H",   -4.62908120,    1.23390028,    0.00000000],
##["C",    2.84394959,   -2.43228059,   -0.00000000],
##["C",    3.52839157,   -1.24679229,    0.00000000],
##["C",    2.83779428,   -0.00000000,   -0.00000000],
##["H",    3.38312959,   -3.39195178,   -0.00000000],
##["H",    4.62908120,   -1.23390028,    0.00000000],
##["C",    3.52839157,    1.24679229,   -0.00000000],
##["C",    2.84394959,    2.43228059,    0.00000000],
##["C",    1.41889714,    2.45760193,   -0.00000000],
##["H",    4.62908120,    1.23390028,   -0.00000000],
##["H",    3.38312959,    3.39195178,    0.00000000],
##["C",   -0.68444198,    3.67907288,    0.00000000],
##["H",   -1.24595161,    4.62585206,    0.00000000],
##["C",    0.68444198,    3.67907288,   -0.00000000],
##["H",    1.24595161,    4.62585206,   -0.00000000]]


#### O3
##a = [
##["O",  0.     ,  0.    ,   0.    ], 
##["O",  0.     ,  1.0885,   0.6697],
##["O",  0.     , -1.0885,   0.6697]]


#### additional test 1, CH4
##a = [
##["C",   0.38000200,  -0.60132300,   0.00000000],
##["H",   0.73665643,  -1.61013300,   0.00000000],
##["H",   0.73667484,  -0.09692481,   0.87365150],
##["H",   0.73667484,  -0.09692481,  -0.87365150],
##["H",  -0.68999800,  -0.60130982,   0.00000000]]


#### additional test 2, CH4
##a = [
##["C",  -0.79222718,  -0.63527652,   0.00000000],
##["H",  -0.43557275,  -1.64408653,   0.00000000],
##["H",  -0.43555434,  -0.13087833,   0.87365150],
##["H",  -0.43555434,  -0.13087833,  -0.87365150],
##["H",  -1.86222718,  -0.63526334,   0.00000000]]


#### additional test 3, CH4
##a = [
##["C",   0.00000000,   0.00000000,   0.00000000],
##["H",   0.00000000,   0.00000000,   1.06999995],
##["H",   0.00000000,  -1.00880563,  -0.35666665],
##["H",  -0.87365130,   0.50440282,  -0.35666665],
##["H",   0.87365130,   0.50440282,  -0.35666665]]


#### additional test 3, H+(H2O)3
##a = [
##[1., -3.2613969123, -0.6233043579,  0.6815344812],
##[8., -2.3118608248,  0.1524013963,  0.3721500130],
##[1., -2.4249335008,  1.1246183202,  0.5593911943],
##[1., -1.3879971070, -0.1199181966,  0.5339250450],
##[8., -4.1314389181, -1.3181125941,  0.9221572855],
##[1., -4.5674163396, -1.8124710340,  0.1092317735],
##[1., -4.8504251361, -0.9024403801,  1.4286443588],
##[8., -2.5784277579,  2.7229822690,  0.6792296664],
##[1., -2.0669892287,  3.2901453187,  1.2682654734],
##[1., -2.6225421443,  3.1817005533, -0.1780402515],
##[8., -5.2320263518, -2.6028173405, -0.9795639123],
##[1., -5.2438622059, -3.5703373453, -1.0121644518],
##[1., -5.2212197080, -2.2977914822, -1.8921589996]]


#### additional test 4, C60
##a = [
##["C",   -0.67264571,   -0.47085201,    0.00000000],
##["C",    0.79124329,   -0.47085201,    0.00000000],
##["C",    1.48353229,    0.72834599,    0.00000000],
##["C",    0.75155029,    1.99610399,   -0.00001900],
##["C",   -0.63311771,    1.99607199,    0.00000500],
##["C",   -1.36502371,    0.72827699,    0.00002100],
##["C",   -1.12502571,   -1.57723701,   -0.84514300],
##["C",    0.05926029,   -2.26098901,   -1.36751300],
##["C",    1.24362029,   -1.57719701,   -0.84518700],
##["C",    2.36379129,   -1.42449601,   -1.64467700],
##["C",    2.66778729,    0.88976899,   -0.84525000],
##["C",    1.48343629,    2.94102799,   -0.84526500],
##["C",    0.79103329,    3.83480599,   -1.64470200],
##["C",   -0.67285971,    3.83478299,   -1.64464800],
##["C",   -1.36512571,    2.94097899,   -0.84515400],
##["C",   -2.54944771,    2.25717199,   -1.36745400],
##["C",   -2.54938571,    0.88965499,   -0.84510200],
##["C",   -2.97729171,   -0.15684501,   -1.64450700],
##["C",   -2.24529471,   -1.42460401,   -1.64452900],
##["C",    0.05920829,   -2.75509401,   -2.66101900],
##["C",   -1.12511771,   -2.59372601,   -3.50614400],
##["C",   -2.24533171,   -1.94696801,   -3.01203600],
##["C",   -2.97734571,   -1.00206001,   -3.85720300],
##["C",   -3.42971571,    0.10429199,   -3.01201000],
##["C",   -3.42976771,    1.39780999,   -3.50610800],
##["C",   -2.97740171,    2.50418499,   -2.66095700],
##["C",   -2.24551671,    3.44914799,   -3.50619100],
##["C",   -1.12528971,    4.09595099,   -3.01217300],
##["C",    2.66772629,    2.25730299,   -1.36763700],
##["C",    1.24329029,    3.07948799,   -5.67325800],
##["C",    2.36356029,    2.92686599,   -4.87388600],
##["C",    3.09553329,    1.65911499,   -4.87388000],
##["C",    2.66763129,    0.61261999,   -5.67330600],
##["C",    1.48330129,    0.77399899,   -6.51843700],
##["C",   -0.67298171,    1.97310399,   -6.51837200],
##["C",   -1.12535071,    3.07946299,   -5.67318700],
##["C",    0.05895829,    3.76326399,   -5.15088600],
##["C",    0.05899529,    4.25736199,   -3.85737800],
##["C",    1.24335429,    4.09596699,   -3.01221300],
##["C",    2.36358029,    3.44922999,   -3.50632700],
##["C",    3.54792829,    1.39796499,   -3.50637300],
##["C",    3.54798529,    0.10447499,   -3.01226300],
##["C",    3.09559229,   -1.00192601,   -3.85740200],
##["C",    2.66766429,   -0.75490501,   -5.15092200],
##["C",    1.48338029,   -1.43870101,   -5.67323000],
##["C",    0.75137829,   -0.49380101,   -6.51838300],
##["C",   -0.63329171,   -0.49381701,   -6.51832700],
##["C",   -1.36527871,    0.77393299,   -6.51833200],
##["C",   -2.24555171,    2.92676999,   -4.87370400],
##["C",   -2.97745071,    1.65897699,   -4.87365800],
##["C",   -2.54956571,    0.61250099,   -5.67310300],
##["C",   -2.54951071,   -0.75502701,   -5.15073900],
##["C",   -1.36518371,   -1.43877001,   -5.67312600],
##["C",   -0.67279571,   -2.33254901,   -4.87371500],
##["C",    0.79109729,   -2.33252301,   -4.87374700],
##["C",    1.24351229,   -2.59365501,   -3.50626500],
##["C",    2.36373329,   -1.94685601,   -3.01222400],
##["C",    3.09555529,    2.50434699,   -2.66118400],
##["C",    3.09567029,   -0.15669501,   -1.64469900],
##["C",    0.79092629,    1.97314199,   -6.51840400]]
###############################################################################



if __name__ == "__main__":
    mol_type = main(a)
    print("The molecular is {:}\n".format(mol_type))
##    mol_type, count = main(a)
##    print("The molecular is {:}, and has {:} reflection\n".format(mol_type, count))


