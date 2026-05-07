from rdkit import Chem
from pathlib import Path
import sys

def remove_zero_coordinate_molecules(input_file):
    """
    Remove molecules from SDF file where all atomic coordinates are zero.
    This typically indicates failed 3D coordinate generation.
    """
    #print('Reading the molecules...')
    suppl = Chem.SDMolSupplier(input_file)

    # Check if every molecule was correctly converted
    read_errors = 0
    for mol in suppl:
        try:
            if mol is not None:
                mol.GetNumAtoms()
        except Exception as e:
            read_errors += 1
    if read_errors>0:
        print(f'{read_errors} molecules could not be read due to errors...')

    # Reset supplier and create list of valid molecules
    suppl = Chem.SDMolSupplier(input_file)
    mols = [mol for mol in suppl if mol is not None]

    # Identify molecules with all zero coordinates
    zero_coord_names = []
    valid_mols = []
    
    for mol in mols:
        try:
            conformer = mol.GetConformer()
            num_atoms = mol.GetNumAtoms()
            
            # Check if all atoms have zero coordinates
            all_zero = True
            for i in range(num_atoms):
                pos = conformer.GetAtomPosition(i)
                if not (abs(pos.x) < 1e-6 and abs(pos.y) < 1e-6 and abs(pos.z) < 1e-6):
                    all_zero = False
                    break
            
            if all_zero:
                mol_name = mol.GetProp('_Name') if mol.HasProp('_Name') else f'Unknown_{len(zero_coord_names)}'
                zero_coord_names.append(mol_name)
            else:
                valid_mols.append(mol)
                
        except Exception as e:
            continue
    
    if len(zero_coord_names)>0:
        print(f'Removing {len(zero_coord_names)} molecules with zero coordinates')
        print(f'{len(valid_mols)} molecules remain after filtering')
        
    else:
        print('All molecules passed the validity check')
    
    if len(valid_mols) == 0:
        print("Warning: No valid molecules remain!")
        return
    
    # Create output file
    output_file = f'{Path(input_file).stem}_no_nulls.sdf'
    
    # Write valid molecules to new SDF file
    writer = Chem.SDWriter(output_file)
    for mol in valid_mols:
        writer.write(mol)
    writer.close()
    
    print(f'Clean SDF file saved as: {output_file}')
    
    return zero_coord_names

def validate_coordinates(input_file):
    """
    Quick validation function to check coordinate distribution in SDF file.
    """
    suppl = Chem.SDMolSupplier(input_file)
    
    coord_stats = {
        'total_molecules': 0,
        'zero_coord_molecules': 0,
        'valid_molecules': 0,
        'read_errors': 0
    }
    
    for mol in suppl:
        coord_stats['total_molecules'] += 1
        
        if mol is None:
            coord_stats['read_errors'] += 1
            continue
            
        try:
            conformer = mol.GetConformer()
            num_atoms = mol.GetNumAtoms()
            
            all_zero = True
            for i in range(num_atoms):
                pos = conformer.GetAtomPosition(i)
                if not (abs(pos.x) < 1e-6 and abs(pos.y) < 1e-6 and abs(pos.z) < 1e-6):
                    all_zero = False
                    break
            
            if all_zero:
                coord_stats['zero_coord_molecules'] += 1
            else:
                coord_stats['valid_molecules'] += 1
                
        except Exception:
            coord_stats['read_errors'] += 1
    
    return coord_stats


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python this-pythonscript.py <input_file>")
        sys.exit(1)
    input_file = sys.argv[1]

    stats = validate_coordinates(input_file)
    
    print("\n=== Cleaning File ===")
    # Clean the file
    removed_names = remove_zero_coordinate_molecules(input_file)


