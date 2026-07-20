
from atomm import Atom

class Receptor :
    def __init__(self, fileReceptorPdbqt) :
        self.allatoms = []
    
        pf = open(fileReceptorPdbqt, "r")
        receptorLines = pf.readlines()
        pf.close()

        for i in range( 0, len(receptorLines) ) :
            line = receptorLines[i]
            if line[0:6] == "ATOM  " or line[0:6] == "HETATM" :
                a1 = Atom(line)
                self.allatoms.append(a1)
                
    def __str__(self) :  # used for print()
        str01 = ""
        for a2 in self.allatoms:
            str01 += a2.__str__() + "\n"
        return str01


if __name__ == "__main__" :        
    
    # debug
    rec = Receptor("../test/1a30_protein.pdbqt")
    
    print(rec)
