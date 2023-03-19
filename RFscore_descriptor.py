# RF-Score_desc.c (all rights reserved): This program reads a list of      
#                PDBbind protein-ligand complexes and calculate their     
#                 RF-Score descriptors                                     
#                                                                          
# Author:        Dr Pedro J. Ballester                                                                                                               
# Purpose:       Preprocessing PBDbind protein-ligand complexes                                                                                     
# Usage:         Read Appendix_A1.doc                                     
# Rewrite of RF-Score_desc.c in python

import os
import numpy as  np

NATOMAX = 3500   # maximum number of heavy atoms in the bindsite
NLIGATMAX = 250 # maximum number of heavy atoms in the ligand
NELEMTS = 54   # maximum number of chemical elements considered
DCUTOFF = 12   # distance cutoff for protein atoms near ligand
VERBOSE = 1  # show information about run

atomic_number_to_name = {
    6 : ["C" , "CA" , "CB" , "CD" , "CD1" , "CD2" , "CE" , "CE1" , 
                    "CE2", "CE3", "CG", "CG1", "CG2", "CH2", "CZ", "CZ2", "CZ3"],
    8 : ["O" , "OD1" , "OD2" , "OE1" , "OE1A" , "OE1B" , "OE2" , "OG" , "OG1", "OH", "OXT"],
    7 : ["N" , "NE" , "NE1" , "NE2" , "NE2A" , "NE2B" , "ND1" , "ND2" , "NH1" , "NH2" , "NZ"],
    9 : ["F"],
    15 : ["P"],
    16 : ["S" , "SD" , "SG"],
    17 : ["Cl"],
    35 : ["Br"],
    53 : ["I"],
}

name_to_atomic_number = { }
for k, v in atomic_number_to_name.items():
    for i in v:
        name_to_atomic_number[i] = k


class PdbLine:
    def __init__(self, input):
        self.lineid = input[0].strip()
        self.atomid = int(input[1].strip())
        self.atomtype = input[2].strip()
        self.resname = input[3].strip()
        self.chainid = input[4].strip()
        self.resid = int(input[5].strip())
        self.r = [float(i.strip()) for i in input[6:9]]
        self.prob = float(input[9].strip())
        self.dum = float(input[10].strip())
        self.atomname = input[11].strip()
    
    def set_atomnumber(self, a):
        self.atomnumber = a

class Ligand:
    def __init__(self):
        self.natoms = 0
        self.atomid = [0] * NATOMAX
        self.atomtype = [" " * 5] * NATOMAX
        self.protname = ""
        self.chainid = ""
        self.resid = 0
        self.r = [[0.0] * 3] * NATOMAX
        self.atomname = [" " * 4] * NATOMAX
        self.atomnumber = [0] * NATOMAX
        self.bindaff = 0.0

def read_pdb_file(filename):
    pdb_lines = []
    with open(filename, "r") as f:
        for line in f.readlines():
            if line.startswith("ATOM"):
                indices = [0, 6, 12, 17, 20, 22, 26, 30, 38, 46, 54, 60, 66, 78]
                items = [line.strip()[i:j] for i, j in zip(indices, indices[1:]+[None])]
                items.pop(6)
                pdb_line = PdbLine(items)
                atomnumber = name_to_atomic_number.get(pdb_line.atomtype)
                if atomnumber is None:
                    continue
                pdb_line.set_atomnumber(atomnumber)
                pdb_lines.append(pdb_line)
    return pdb_lines

def read_ligand_sdf(filename, bind_dict=None):
    ligand = Ligand()
    with open(filename, "r") as f:
        lines = f.readlines()
        ligand.natoms = int(lines[3][:3].strip())
        if ligand.natoms > NLIGATMAX:
            # then correct reading error
            ligand.natoms = int(line[3][:2].strip())
        ligand.protname = lines[0].split("_")[0].strip()
        for i in range(ligand.natoms):
            line = lines[i+4].split()
            atomnumber = name_to_atomic_number.get(line[3])
            if atomnumber is None:
                continue
            ligand.atomid[i] = i+1
            ligand.atomtype[i] = line[3]
            ligand.r[i][0] = float(line[0])
            ligand.r[i][1] = float(line[1])
            ligand.r[i][2] = float(line[2])
            ligand.atomnumber[i] = atomnumber
        if bind_dict:
            ligand.bindaff = bind_dict[ligand.protname]
        return ligand   
    
def read_bindaff(filename):
    data = np.loadtxt(filename, dtype={'names': ('pdbid', 'affinity'),
                     'formats': ('U4', 'f4')})
    return {data[i][0] : [data[i][1]] for i in range(data.shape[0])}
            

# Main function
def main(dat_file, struct_dir, output=None):
    if output is None:
        output = dat_file.split(".")[0] + "_desc.npz"
    bindaff_dict = read_bindaff(dat_file)
    for prot_name in bindaff_dict:
        pname_lc = prot_name.lower()
        ligand_file = os.path.join(struct_dir, pname_lc, pname_lc +"_ligand.sdf")
        pocket_file = os.path.join(struct_dir, pname_lc, pname_lc +"_pocket.pdb")
        if not os.path.exists(ligand_file):
            if VERBOSE: print(prot_name, "does not exists in dir", struct_dir)
            continue
        if not os.path.exists(pocket_file):
            if VERBOSE: print(prot_name, "pocket does not exists in dir", struct_dir, "; read protein file instead")
            pocket_file = os.path.join(struct_dir, pname_lc, pname_lc +"_protein.pdb")
            if not os.path.exists(pocket_file):
                if VERBOSE: print(prot_name, "does not exists in dir", struct_dir)
                continue

        ligand = read_ligand_sdf(ligand_file)
        pocket = read_pdb_file(pocket_file)
        features = np.zeros((NELEMTS, NELEMTS))

        # Calculate distances between current ligand and its binding site
        break_loop = False
        visited = set()
        for k in range(ligand.natoms):
            for l in range(len(pocket)):
                ddum = np.linalg.norm(np.array(ligand.r[k]) - np.array(pocket[l].r))
                if ddum < DCUTOFF:
                    features[pocket[l].atomnumber, ligand.atomnumber[k]] += 1
                    visited.add(l)
                    if len(visited) >= NATOMAX:
                        break_loop = True
                        if VERBOSE:
                            print("exceed maximum binding site heavy atom limit")
                        break
            if break_loop:
                break
        col_mask = list(atomic_number_to_name.keys())
        bindaff_dict[prot_name].append(features[col_mask][:, col_mask])

    # reorder output
    new_dict = {"pdbid": list(bindaff_dict.keys()),
                "affinity": np.array([bindaff_dict[key][0] for key in bindaff_dict]),
                "feature": [bindaff_dict[key][1] if len(bindaff_dict[key]) > 1 else None
                            for key in bindaff_dict]}

    # save output
    np.savez(output, **new_dict)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        prog="RF-Score_descriptor",
        description="Calculates intermolecular interaction features for RF-Score",
        epilog="A machine learning approach to predicting protein-ligand binding affinity with applications to molecular docking. \
            P.J. Ballester & J.B.O. Mitchell, Bioinformatics, 26, 1169-1175 (2010)"
    )
    parser.add_argument("dat", help="data file containing pdbid and affinity")
    parser.add_argument("dir", help="path to directory that stores all pdbid folders containing protein, ligand, pocket files")
    parser.add_argument("-o", dest="out", default=None, help="path to save output")

    args = parser.parse_args()
    main(args.dat, args.dir, args.out)





