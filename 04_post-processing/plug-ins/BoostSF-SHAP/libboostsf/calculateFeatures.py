
import os 
from ligandm import Ligand
from receptorm import Receptor
import subprocess
import re


def cal36features(fileReceptor, fileLigand) :
    rec = Receptor(fileReceptor)
    lig = Ligand(fileLigand)
    
    reVal = [0.0 for i in range(0, 36)]
    
    for la in lig.allatoms :
        if la.numType == 1000:
            continue
            
        for ra in rec.allatoms :
            if ra.numType == 1000 :
                continue
        
            d2 = (la.coord_x - ra.coord_x)**2 +  (la.coord_y - ra.coord_y)**2 +  (la.coord_z - ra.coord_z)**2
            
            if d2 >= 12*12 :  # RF-Score cutoff 12A
                continue

            reVal[la.numType*4 + ra.numType]  += 1
    
    return reVal

##################################################################################################################################################################################
def calVina5terms(fileReceptor, fileLigand, fileReference)  :
    
    reVal = [0.0 for i in range(0, 5)]
    cmd = [f'smina '
           f'-r {fileReceptor} '
           f'--autobox_ligand {fileReference} '
           f'--autobox_add 6 '
           f'-l {fileLigand} '
           f'--score_only > smina_output.txt']
    #print(cmd)
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    with open('smina_output.txt', 'r') as f:
        lines = []
        for line in f:
            if re.search('##', line):
                lines.append(line)
    terms = lines[1].split(' ')
    reVal = [0.0 for i in range(0, 5)]
    reVal[0] = float(terms[2])
    reVal[1] = float(terms[3])
    reVal[2] = float(terms[4])
    reVal[3] = float(terms[5])
    reVal[4] = float(terms[6])

            
    return reVal  
    
##################################################################################################################################################################################



def cal41features(fileReceptor, fileLigand, fileReference) :
    res1 = cal36features(fileReceptor, fileLigand)
    res2 = calVina5terms(fileReceptor, fileLigand, fileReference)
    
    return res1 + res2
    

if __name__ == "__main__" :        
    
    # ----- debug 1 -----
    # Feas = cal36features("../test/1a30_protein.pdbqt", "./1a30_ligand.pdbqt")
    # print("1a30: ")
    # for t in Feas :
        # print("%.4f,"%t, end="" )

    # ----- debug 2 -----
    # Feas = calVina5terms("../test/1a30_protein.pdbqt", "./1a30_ligand.pdbqt")
    # print("1a30: ")
    # for t in Feas :
        # print("%.4f,"%t, end="" )

    # ----- debug 3 -----   
    Feas = cal41features("../test/d4_prot.pdbqt", "../test/first_lig_pdbqt.pdbqt", "../test/d4_lig.pdbqt")
    for t in Feas :
        print("%.4f,"%t, end="" )
    