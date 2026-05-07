
import numpy as np
import pandas as pd
import argparse

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 

from libboostsf import calculateFeatures

# command-line arg parse
dirname, filename = os.path.split( __file__ )
parser = argparse.ArgumentParser( prog = 'python %s'%filename,
                                  description = "This program is used to extract features from a set of protein-ligand complexes and generate tabular data." )
parser.add_argument("--rllist", required=True, help="The path of the list file, which contains the pathes of receptor and ligand files.")
args = parser.parse_args()


Xpd = pd.DataFrame(columns="C/C,N/C,O/C,S/C,C/N,N/N,O/N,S/N,C/O,N/O,O/O,S/O,C/S,N/S,O/S,S/S,C/P,N/P,O/P,S/P,C/F,N/F,O/F,S/F,C/Cl,N/Cl,O/Cl,S/Cl,C/Br,N/Br,O/Br,S/Br,C/I,N/I,O/I,S/I,gauss1(inter),gauss2(inter),repulsion(inter),hydrophobic(inter),Hydrogen(inter)".split(',') )
    
pfrl = open( args.rllist , "r")
rllistLines = pfrl.readlines()
pfrl.close()

verifiedLines = []
for rlLine in rllistLines :
    if rlLine.lower().count(".pdbqt") == 2 :  # confirm two pdbqt file.
        verifiedLines.append( "  %s"%(rlLine.strip()) )
        print( ("Processing  %s"%rlLine).rstrip() )
        
        fileList = rlLine.split(";")
        rFile = fileList[0].strip()  # receptor file
        lFile = fileList[1].strip()  # ligand file
        
        features41 = calculateFeatures.cal41features(rFile, lFile )
        Xpd.loc[ len(Xpd.index) ] = features41  # append as a new row.


outfile = "prepare-features-file-OUT.csv"
Xsave = Xpd.copy(deep=True)
Xsave['receptor-ligand'] = verifiedLines  # add a comment column
Xsave.to_csv(outfile, float_format='%10.4f', index=False)
print("\nThe generated file is %s"%outfile )
