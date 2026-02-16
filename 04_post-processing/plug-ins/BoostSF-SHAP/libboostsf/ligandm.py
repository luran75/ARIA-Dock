
from atomm import Atom

class Ligand :
    def __init__(self, fileLigandPdbqt) :
        self.allatoms = []
    
        pf = open(fileLigandPdbqt, "r")
        ligandLines = pf.readlines()
        pf.close()

        for i in range( 0, len(ligandLines) ) :
            line = ligandLines[i]
            if line[0:6] == "ATOM  " or line[0:6] == "HETATM" :
                a1 = Atom(line)
                self.allatoms.append(a1)
                
            if line[0:6] == "TORSDO" :
                break 

    def __str__(self) :  # used for print()
        str01 = ""
        for a2 in self.allatoms:
            str01 += a2.__str__() + "\n"
        return str01


if __name__ == "__main__" :        
    
    # debug
    lig = Ligand("../test/1a30_ligand.pdbqt")
    
    print(lig)
