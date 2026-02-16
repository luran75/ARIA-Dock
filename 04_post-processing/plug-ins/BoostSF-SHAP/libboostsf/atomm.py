
atomType = {
#   PDBQT AutoDock atom name : RF-score type ,   # atom name  
    "C"  :                      0            ,   # C  
    "A"  :                      0            ,   # C  
    "N"  :                      1            ,   # N  
    "NA" :                      1            ,   # N  
    "OA" :                      2            ,   # O  
    "S"  :                      3            ,   # S  
    "SA" :                      3            ,   # S  
    "P"  :                      4            ,   # P  
    "F"  :                      5            ,   # F  
    "Cl" :                      6            ,   # Cl 
    "Br" :                      7            ,   # Br 
    "I"  :                      8                # I  
#   the other aotms are         1000
}

class Atom :
    def __init__(self, strAtomRecord) :
        self.coord_x = float( strAtomRecord[30: 38] )
        self.coord_y = float( strAtomRecord[38: 46] )
        self.coord_z = float( strAtomRecord[46: 54] )
        
        str1 = strAtomRecord[77:]
        str1 = str1.rstrip()            # remove space
        if str1 in atomType :
            self.numType = atomType[str1]   
        else :  # not find key 
            self.numType = 1000
        
    def __str__(self) :  # used for print()
        return "coord_x: %.3f, coord_y: %.3f, coord_z: %.3f, numType: %d"%(self.coord_x, self.coord_y, self.coord_z, self.numType)


if __name__ == "__main__" :
    
    # debug
    a = Atom("ATOM     17  NE2 GLN A   2      29.636  36.948  -2.523  1.00 -0.94    -0.370 N \r\n")
    print(a)

    b = Atom("ATOM   1368  O   GLY B  48      14.936  17.093   6.933  1.00 -0.57    -0.272 Fe\r\n")
    print(b)
    