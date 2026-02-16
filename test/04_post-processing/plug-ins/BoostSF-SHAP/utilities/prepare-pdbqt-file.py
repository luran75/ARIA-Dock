
# Run on windows
# dependency:  must install AutoDockTools

import os
import glob
import shutil
import argparse


# command-line arg parse
dirname, filename = os.path.split( __file__ )
parser = argparse.ArgumentParser( prog = 'python %s'%filename,
                                  description = "This program is used to format a protein and a ligand structure files as PDBQT files.")

parser.add_argument("--receptor", required=True, help="The path of the receptor file, 'pdb' file format.")
parser.add_argument("--ligand",   required=True, help="The path of the ligand file, 'mol2' or 'pdb' file format.")
parser.add_argument("--outfolder",    help="The folder to store *.pdbqt files.")
args = parser.parse_args()

in_receptor = args.receptor
in_ligand = args.ligand

assert (in_ligand[-5:]==".mol2" or in_ligand[-4:]==".pdb"), "The ligand file should be 'mol2' or 'pdb' file format."

if args.outfolder :
    out_folder = args.outfolder
else: 
    out_folder = "./"


# --- receptor ---
python_location = "C:/Progra~2/MGLTools-1.5.6/python  "
pre_rec = "C:/Progra~2/MGLTools-1.5.6/Lib/site-packages/AutoDockTools/Utilities24/prepare_receptor4.py  "

op = "-r   %s  -U nphs_lps_waters_deleteAltB "%(in_receptor)

os.system( python_location + pre_rec + op )


# --- ligand ---
python_location = "C:/Progra~2/MGLTools-1.5.6/python  "
pre_lig = "C:/Progra~2/MGLTools-1.5.6/Lib/site-packages/AutoDockTools/Utilities24/prepare_ligand4.py  "

op = "-l   %s  "%(in_ligand)

os.system( python_location + pre_lig + op )


# replace the \r\n as \n, important for linux compatibility software
head_tail = os.path.split(in_ligand)
name, suffix = head_tail[1].split('.')
fnLigand = name + ".pdbqt"

FileLig = open("./%s"%fnLigand, "r")
lines = FileLig.readlines()
FileLig.close()
FileLig = open("./%s"%fnLigand, "wb")
for line in lines :
    FileLig.write( line.encode('ascii') )
FileLig.close()


# --- move file  ---
if args.outfolder :
    pdbqtfiles = glob.glob("*.pdbqt")
    for onefile in pdbqtfiles :
        shutil.move( onefile, out_folder ) 

