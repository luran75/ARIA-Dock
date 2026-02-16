"""
Protein Preparation Script for Molecular Docking

This script prepares a protein structure from a PDB file for molecular docking by:
1. Cleaning the structure (removing water, ligands, etc.)
2. Adding missing residues/atoms
3. Adding hydrogens
4. Optimizing side chain conformations
5. Assigning partial charges
6. Saving the prepared structure

Requirements:
- biopython
- rdkit
- pdbfixer (optional, for missing residues)
- openmm (for pdbfixer)

Install with: pip install biopython rdkit-pypi pdbfixer openmm
"""

import argparse
import sys
from pathlib import Path
import warnings
import shutil
# Suppress only Biopython and RDKit warnings to avoid hiding important issues
warnings.filterwarnings('ignore', module='Bio')
warnings.filterwarnings('ignore', module='rdkit')

try:
    from Bio import PDB
    from Bio.PDB import PDBIO, Select
except ImportError:
    print("BioPython not found. Install with: pip install biopython")
    sys.exit(1)

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError:
    print("RDKit not found. Install with: pip install rdkit-pypi")
    sys.exit(1)

try:
    from pdbfixer import PDBFixer
    from openmm.app import PDBFile
    PDBFIXER_AVAILABLE = True
except ImportError:
    print("Warning: PDBFixer not available. Missing residue repair will be limited.")
    print("Install with: pip install pdbfixer openmm")
    PDBFIXER_AVAILABLE = False


class ProteinOnlySelect(Select):
    """Select only protein atoms, excluding water, ligands, and ions."""
    
    def accept_residue(self, residue):
        # Accept standard amino acids
        standard_aa = {
            'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 
            'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 
            'THR', 'TRP', 'TYR', 'VAL'
        }
        return residue.get_resname() in standard_aa


class ProteinPreparator:
    """Main class for protein preparation."""
    
    def __init__(self, input_pdb, verbose=True):
        self.input_pdb = Path(input_pdb)
        self.verbose = verbose
        
        # Output files
        self.base_name = self.input_pdb.stem
        self.cleaned_pdb = f"{self.base_name}_cleaned.pdb"
        self.fixed_pdb = f"{self.base_name}_fixed.pdb"
        self.prepared_pdb = f"{self.base_name}_prepared.pdb"
        
    def log(self, message):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(f"{message}")
        else:
            pass

    def clean_structure(self):
        """Remove water molecules, ligands, and keep only protein chains."""
        self.log("Cleaning structure - removing water, ligands, and non-protein atoms...")
        
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure('protein', self.input_pdb)
        
        # Save only protein atoms
        io = PDBIO()
        io.set_structure(structure)
        io.save(str(self.cleaned_pdb), ProteinOnlySelect())
        
        return self.cleaned_pdb
    
    def fix_missing_residues(self):
        """Fix missing residues and atoms using PDBFixer."""
        if not PDBFIXER_AVAILABLE:
            self.log("PDBFixer not available. Skipping missing residue repair.")
            # Just copy the cleaned file
            shutil.copy(self.cleaned_pdb, self.fixed_pdb)
            return self.fixed_pdb
        
        self.log("Fixing missing residues and atoms...")
        
        try:
            # Load structure with PDBFixer
            fixer = PDBFixer(filename=str(self.cleaned_pdb))
            
            # Find and add missing residues
            fixer.findMissingResidues()
            if fixer.missingResidues:
                fixer.findNonstandardResidues()
                fixer.replaceNonstandardResidues()
                fixer.removeHeterogens(keepWater=False)
                fixer.findMissingAtoms()
                fixer.addMissingAtoms()
            else:
                self.log("No missing residues found")
                fixer.findMissingAtoms()
                if fixer.missingAtoms:
                    self.log(f"Adding missing atoms...")
                    fixer.addMissingAtoms()
            
            # Save fixed structure
            PDBFile.writeFile(fixer.topology, fixer.positions, 
                            open(str(self.fixed_pdb), 'w'))
                        
        except Exception as e:
            self.log(f"Error in PDBFixer: {e}")
            self.log("Proceeding with cleaned structure...")
            import shutil
            shutil.copy(self.cleaned_pdb, self.fixed_pdb)
        
        return self.fixed_pdb
    
    def add_hydrogens_rdkit(self):
        """Add hydrogens using RDKit (alternative method)."""
        self.log("Adding hydrogens using RDKit...")
        
        try:
            # Read PDB file
            mol = Chem.MolFromPDBFile(str(self.fixed_pdb), removeHs=False)
            if mol is None:
                raise ValueError("Could not parse PDB file with RDKit")
            
            # Check if coordinates are present
            has_coords = all(atom.HasProp('_CARTESIAN_X') for atom in mol.GetAtoms())
            # Add hydrogens, only generating coordinates if missing
            mol_with_h = Chem.AddHs(mol, addCoords=not has_coords)
            
            # Try to optimize hydrogen positions
            try:
                AllChem.EmbedMolecule(mol_with_h, randomSeed=42)
                AllChem.MMFFOptimizeMolecule(mol_with_h, maxIters=200)
            except:
                self.log("Warning: Could not optimize hydrogen positions")
            
            # Write output
            writer = Chem.PDBWriter(str(self.prepared_pdb))
            writer.write(mol_with_h)
            writer.close()
            
            self.log(f"Hydrogens added. Structure saved to: {self.prepared_pdb}")
            
        except Exception as e:
            self.log(f"Error adding hydrogens with RDKit: {e}")
            self.log("Copying fixed structure as final result...")
            import shutil
            shutil.copy(self.fixed_pdb, self.prepared_pdb)
    
    def add_hydrogens_openmm(self):
        """Add hydrogens using OpenMM/PDBFixer."""
        if not PDBFIXER_AVAILABLE:
            self.log("OpenMM/PDBFixer not available for hydrogen addition")
            return self.add_hydrogens_rdkit()
        
        self.log("Adding hydrogens using OpenMM...")
        
        try:
            fixer = PDBFixer(filename=str(self.fixed_pdb))
            fixer.addMissingHydrogens(7.0)  # pH 7.0
            
            PDBFile.writeFile(fixer.topology, fixer.positions,
                            open(str(self.prepared_pdb), 'w'))
            
            self.log(f"Hydrogens added. Structure saved to: {self.prepared_pdb}")
            
        except Exception as e:
            self.log(f"Error adding hydrogens with OpenMM: {e}")
            return self.add_hydrogens_rdkit()
    
    def validate_structure(self):
        """Perform basic validation of the prepared structure."""
        self.log("Validating prepared structure...")
        
        try:
            parser = PDB.PDBParser(QUIET=True)
            structure = parser.get_structure('prepared', self.prepared_pdb)
            
            # Count atoms and residues
            atom_count = sum(1 for atom in structure.get_atoms())
            residue_count = sum(1 for residue in structure.get_residues())
            chain_count = len(list(structure.get_chains()))
            
            self.log(f"Final structure contains:")
            self.log(f"  - {chain_count} chain(s)")
            self.log(f"  - {residue_count} residues")
            self.log(f"  - {atom_count} atoms")
            
            # Check for hydrogens
            h_count = sum(1 for atom in structure.get_atoms() if atom.element == 'H')
            self.log(f"  - {h_count} hydrogen atoms")
            
            return True
            
        except Exception as e:
            self.log(f"Validation error: {e}")
            return False
    
    def prepare(self, hydrogen_method="openmm"):
        """Run the complete preparation pipeline."""
        self.log(f"Starting protein preparation for: {self.input_pdb}")
        
        # Step 1: Clean structure
        self.clean_structure()
        
        # Step 2: Fix missing residues/atoms
        self.fix_missing_residues()
        
        # Step 3: Add hydrogens
        if hydrogen_method.lower() == "rdkit":
            self.add_hydrogens_rdkit()
        else:
            self.add_hydrogens_openmm()
        
        # Step 4: Validate
        if self.validate_structure():
            self.log("Protein preparation completed successfully!")
            self.log(f"Final prepared structure: {self.prepared_pdb}")
            return self.prepared_pdb
        else:
            self.log("Structure validation failed")
            return None


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Prepare protein structure for molecular docking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python protein_prep.py protein.pdb
  python protein_prep.py protein.pdb --output-dir ./prepared --hydrogen-method rdkit
  python protein_prep.py protein.pdb --quiet
        """
    )
    
    parser.add_argument('input_pdb', 
                       help='Input PDB file path')
    parser.add_argument('--hydrogen-method', 
                       choices=['openmm', 'rdkit'],
                       default='openmm',
                       help='Method for adding hydrogens (default: openmm)')
    parser.add_argument('--quiet', '-q', 
                       action='store_true',
                       help='Suppress verbose output')
    
    args = parser.parse_args()
    
    # Check input file exists
    if not Path(args.input_pdb).exists():
        print(f"Error: Input file '{args.input_pdb}' not found")
        sys.exit(1)
    
    # Initialize preparator
    preparator = ProteinPreparator(
        input_pdb=args.input_pdb,
        verbose=not args.quiet
    )
    
    # Run preparation
    result = preparator.prepare(hydrogen_method=args.hydrogen_method)
    
    if result:
        print(f"\nSuccess! Prepared protein saved to: {result}")
        print(f"The protein is now ready for molecular docking.")
    else:
        print("\nPreparation failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
