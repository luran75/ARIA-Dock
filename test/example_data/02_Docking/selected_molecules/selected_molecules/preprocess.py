"""
Enhanced 3D Molecule Preprocessing Script for Virtual Screening
Converts SMILES to 3D SDF format with comprehensive validation and filtering
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Dict
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen, Lipinski, rdMolDescriptors
from rdkit.Chem import PandasTools
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import RDLogger

# Optional imports
try:
    import datamol as dm
    DATAMOL_AVAILABLE = True
except ImportError:
    DATAMOL_AVAILABLE = False
    print("Warning: datamol not available. Using RDKit-only standardization.")

try:
    from openbabel import openbabel as ob
    OPENBABEL_AVAILABLE = True
except ImportError:
    OPENBABEL_AVAILABLE = False
    print("Warning: OpenBabel not available. pH adjustment disabled.")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')
ob.obErrorLog.StopLogging()

class MoleculeProcessor:
    """Main class for processing molecules from SMILES to 3D structures."""
    
    # Valid atoms for drug-like molecules (expanded set)
    VALID_ATOMS = {'C', 'H', 'O', 'N', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'B', 'Si'}
    
    # Drug-likeness filters (Lipinski's Rule of Five + extensions)
    LIPINSKI_LIMITS = {
        'mw_max': 500,
        'logp_max': 5,
        'hbd_max': 5,
        'hba_max': 10,
        'rotatable_bonds_max': 10,
        'tpsa_max': 140,
        'aromatic_rings_max': 5
    }
    
    def __init__(self, apply_lipinski: bool = False, strict_atom_filter: bool = True):
        """
        Initialize the molecule processor.
        
        Args:
            apply_lipinski: Apply Lipinski's Rule of Five filtering
            strict_atom_filter: Use strict atom filtering for drug-like molecules
        """
        self.apply_lipinski = apply_lipinski
        self.strict_atom_filter = strict_atom_filter
        self.salt_remover = SaltRemover()
        self.standardizer = rdMolStandardize.Normalizer()
        
        # Statistics tracking
        self.stats = {
            'input_molecules': 0,
            'duplicates_removed': 0,
            'invalid_smiles': 0,
            'invalid_atoms': 0,
            'lipinski_violations': 0,
            'salt_removal_failed': 0,
            'standardization_failed': 0,
            '3d_embedding_failed': 0,
            'final_molecules': 0
        }
        
        # Track failed molecules for debugging
        self.failed_molecules = {
            'invalid_smiles': [],
            'invalid_atoms': [],
            'lipinski_violations': [],
            'salt_removal_failed': [],
            'standardization_failed': [],
            '3d_embedding_failed': []
        }
    
    def read_smiles_file(self, filename: str):
        """
        Read SMILES from file with improved error handling.
        
        Args:
            filename: Path to input file
            
        Returns:
            DataFrame with 'smiles' and 'name' columns
        """
        logger.info(f"Reading SMILES from {filename}")
        
        if not Path(filename).exists():
            raise FileNotFoundError(f"Input file {filename} not found")
        
        rows = []
        with open(filename, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Handle different delimiters
                parts = line.replace('\t', ' ').split()
                parts = [p for p in parts if p]  # Remove empty parts
                
                if len(parts) >= 2:
                    smiles, name = parts[0], parts[1]
                    # Basic SMILES validation
                    if self._is_valid_smiles_format(smiles):
                        rows.append({'smiles': smiles, 'name': name, 'line_number': line_num})
                    else:
                        logger.warning(f"Invalid SMILES format at line {line_num}: {smiles}")
                elif len(parts) == 1:
                    # No name provided, use line number
                    smiles = parts[0]
                    if self._is_valid_smiles_format(smiles):
                        rows.append({'smiles': smiles, 'name': f"mol_{line_num}", 'line_number': line_num})
        
        if not rows:
            raise ValueError("No valid SMILES found in input file")
        
        df = pd.DataFrame(rows)
        self.stats['input_molecules'] = len(df)
        
        # Remove duplicates
        initial_count = len(df)
        df = df.drop_duplicates(subset='smiles', keep='first').reset_index(drop=True)
        self.stats['duplicates_removed'] = initial_count - len(df)
        
        logger.info(f"Loaded {len(df)} unique molecules ({self.stats['duplicates_removed']} duplicates removed)")
        return df
    
    def _is_valid_smiles_format(self, smiles: str):
        """Basic SMILES format validation."""
        if not smiles or len(smiles) < 2:
            return False
        # Skip obvious metal complexes with many dots
        if smiles.count('.') > 1:
            return False
        return True
    
    def standardize_molecules(self, df: pd.DataFrame, protonate: bool = False, 
                            target_ph: float = 7.4):
        """
        Standardize SMILES with multiple approaches.
        
        Args:
            df: DataFrame with SMILES
            protonate: Apply pH-dependent protonation
            target_ph: Target pH for protonation
            
        Returns:
            DataFrame with standardized molecules
        """
        logger.info("Standardizing molecules...")
        
        results = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Standardizing"):
            smiles = row['smiles']
            name = row['name']
            
            try:
                # Convert to molecule
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    self.stats['invalid_smiles'] += 1
                    self.failed_molecules['invalid_smiles'].append({
                        'name': name,
                        'smiles': smiles,
                        'reason': 'Could not parse SMILES string'
                    })
                    continue
                
                # Remove salts
                mol = self._remove_salts(mol)
                if mol is None:
                    self.stats['salt_removal_failed'] += 1
                    self.failed_molecules['salt_removal_failed'].append({
                        'name': name,
                        'smiles': smiles,
                        'reason': 'Salt removal failed'
                    })
                    continue
                
                # Atom filtering
                if self.strict_atom_filter and not self._has_valid_atoms(mol):
                    self.stats['invalid_atoms'] += 1
                    invalid_atoms = [atom.GetSymbol() for atom in mol.GetAtoms() 
                                   if atom.GetSymbol() not in self.VALID_ATOMS]
                    self.failed_molecules['invalid_atoms'].append({
                        'name': name,
                        'smiles': smiles,
                        'reason': f'Invalid atoms detected: {list(set(invalid_atoms))}'
                    })
                    continue
                
                # Standardization
                mol = self._standardize_molecule(mol, protonate, target_ph)
                if mol is None:
                    self.stats['standardization_failed'] += 1
                    self.failed_molecules['standardization_failed'].append({
                        'name': name,
                        'smiles': smiles,
                        'reason': 'Molecule standardization failed'
                    })
                    continue
                
                # Drug-likeness filtering
                if self.apply_lipinski and not self._passes_lipinski(mol):
                    self.stats['lipinski_violations'] += 1
                    violation_details = self._get_lipinski_violations(mol)
                    self.failed_molecules['lipinski_violations'].append({
                        'name': name,
                        'smiles': smiles,
                        'reason': f'Lipinski violations: {violation_details}'
                    })
                    continue
                
                # Store molecule with properties
                mol.SetProp('_Name', name)
                mol.SetProp('original_smiles', smiles)
                
                results.append({
                    'name': name,
                    'mol': mol,
                    'smiles': Chem.MolToSmiles(mol),
                    'original_smiles': smiles
                })
                
            except Exception as e:
                logger.debug(f"Error processing {name}: {e}")
                self.stats['standardization_failed'] += 1
                self.failed_molecules['standardization_failed'].append({
                    'name': name,
                    'smiles': smiles,
                    'reason': f'Processing error: {str(e)}'
                })
                continue
        
        result_df = pd.DataFrame(results)
        logger.info(f"Standardized {len(result_df)} molecules")
        return result_df
    
    def _remove_salts(self, mol: Chem.Mol):
        """Remove salts from molecule."""
        try:
            return self.salt_remover.StripMol(mol)
        except:
            return None
    
    def _has_valid_atoms(self, mol: Chem.Mol) -> bool:
        """Check if molecule contains only valid atoms."""
        if mol is None:
            return False
        
        for atom in mol.GetAtoms():
            if atom.GetSymbol() not in self.VALID_ATOMS:
                return False
        return True
    
    def _standardize_molecule(self, mol: Chem.Mol, protonate: bool, target_ph: float):
        """Apply molecular standardization."""
        try:
            # Use datamol if available
            if DATAMOL_AVAILABLE:
                mol = dm.fix_mol(mol)
                mol = dm.sanitize_mol(mol, sanifix=True, charge_neutral=not protonate)
                
                if protonate:
                    mol = dm.standardize_mol(mol, disconnect_metals=True, normalize=True, 
                                          reionize=False, uncharge=False, stereo=True)
                else:
                    mol = dm.standardize_mol(mol, disconnect_metals=True, normalize=True, 
                                          reionize=True, uncharge=True, stereo=True)
            else:
                # RDKit-only standardization
                mol = self.standardizer.normalize(mol)
                Chem.SanitizeMol(mol)
                if not protonate:
                    mol = rdMolStandardize.Uncharger().uncharge(mol)
            
            # pH adjustment using OpenBabel if available and requested
            if protonate and OPENBABEL_AVAILABLE:
                mol = self._adjust_ph_openbabel(mol, target_ph)
            
            return mol
            
        except Exception as e:
            logger.debug(f"Standardization failed: {e}")
            return None
    
    def _adjust_ph_openbabel(self, mol: Chem.Mol, ph: float):
        """Adjust molecule protonation state using OpenBabel."""
        try:
            smiles = Chem.MolToSmiles(mol)
            
            obc = ob.OBConversion()
            obc.SetInAndOutFormats('smi', 'smi')
            obmol = ob.OBMol()
            obc.ReadString(obmol, smiles)
            
            obmol.CorrectForPH(ph)
            obmol.AddHydrogens(False, True)
            obmol.ConvertDativeBonds()
            
            adjusted_smiles = obc.WriteString(obmol).strip()
            return Chem.MolFromSmiles(adjusted_smiles)
            
        except Exception as e:
            logger.debug(f"pH adjustment failed: {e}")
            return mol  # Return original if adjustment fails
    
    def _passes_lipinski(self, mol: Chem.Mol):
        """Check if molecule passes drug-likeness filters."""
        try:
            mw = Descriptors.MolWt(mol)
            logp = Crippen.MolLogP(mol)
            hbd = Lipinski.NumHDonors(mol)
            hba = Lipinski.NumHAcceptors(mol)
            rotatable = rdMolDescriptors.CalcNumRotatableBonds(mol)
            tpsa = rdMolDescriptors.CalcTPSA(mol)
            aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
            
            violations = 0
            if mw > self.LIPINSKI_LIMITS['mw_max']:
                violations += 1
            if logp > self.LIPINSKI_LIMITS['logp_max']:
                violations += 1
            if hbd > self.LIPINSKI_LIMITS['hbd_max']:
                violations += 1
            if hba > self.LIPINSKI_LIMITS['hba_max']:
                violations += 1
            if rotatable > self.LIPINSKI_LIMITS['rotatable_bonds_max']:
                violations += 1
            if tpsa > self.LIPINSKI_LIMITS['tpsa_max']:
                violations += 1
            if aromatic_rings > self.LIPINSKI_LIMITS['aromatic_rings_max']:
                violations += 1
            
            return violations <= 1  # Allow one violation
            
        except:
            return True  # If calculation fails, don't filter out
    
    def _get_lipinski_violations(self, mol: Chem.Mol):
        """Get detailed information about Lipinski violations."""
        try:
            violations = []
            
            mw = Descriptors.MolWt(mol)
            if mw > self.LIPINSKI_LIMITS['mw_max']:
                violations.append(f"MW={mw:.1f} (>{self.LIPINSKI_LIMITS['mw_max']})")
            
            logp = Crippen.MolLogP(mol)
            if logp > self.LIPINSKI_LIMITS['logp_max']:
                violations.append(f"LogP={logp:.1f} (>{self.LIPINSKI_LIMITS['logp_max']})")
            
            hbd = Lipinski.NumHDonors(mol)
            if hbd > self.LIPINSKI_LIMITS['hbd_max']:
                violations.append(f"HBD={hbd} (>{self.LIPINSKI_LIMITS['hbd_max']})")
            
            hba = Lipinski.NumHAcceptors(mol)
            if hba > self.LIPINSKI_LIMITS['hba_max']:
                violations.append(f"HBA={hba} (>{self.LIPINSKI_LIMITS['hba_max']})")
            
            rotatable = rdMolDescriptors.CalcNumRotatableBonds(mol)
            if rotatable > self.LIPINSKI_LIMITS['rotatable_bonds_max']:
                violations.append(f"RotBonds={rotatable} (>{self.LIPINSKI_LIMITS['rotatable_bonds_max']})")
            
            tpsa = rdMolDescriptors.CalcTPSA(mol)
            if tpsa > self.LIPINSKI_LIMITS['tpsa_max']:
                violations.append(f"TPSA={tpsa:.1f} (>{self.LIPINSKI_LIMITS['tpsa_max']})")
            
            aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
            if aromatic_rings > self.LIPINSKI_LIMITS['aromatic_rings_max']:
                violations.append(f"AromaticRings={aromatic_rings} (>{self.LIPINSKI_LIMITS['aromatic_rings_max']})")
            
            return "; ".join(violations) if violations else "No violations"
            
        except Exception as e:
            return f"Error calculating violations: {str(e)}"
    
    def generate_3d_conformers(self, df: pd.DataFrame, num_conformers: int = 1,
                             optimize: bool = True, random_seed: int = 42):
        """
        Generate 3D conformers for molecules.
        
        Args:
            df: DataFrame with molecules
            num_conformers: Number of conformers to generate
            optimize: Apply MMFF optimization
            random_seed: Random seed for reproducibility
            
        Returns:
            DataFrame with 3D molecules
        """
        logger.info(f"Generating 3D conformers for {len(df)} molecules...")
        
        results = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="3D Embedding"):
            mol = row['mol']
            name = row['name']
            
            try:
                # Add hydrogens
                mol_3d = Chem.AddHs(mol)
                
                # Generate conformer(s)
                success = self._embed_molecule(mol_3d, num_conformers, random_seed)
                
                if success:
                    if optimize:
                        self._optimize_conformers(mol_3d)
                    
                    # Copy properties
                    for prop in mol.GetPropNames():
                        mol_3d.SetProp(prop, mol.GetProp(prop))
                    
                    results.append({
                        'name': name,
                        'mol': mol_3d,
                        'smiles': row['smiles'],
                        'original_smiles': row.get('original_smiles', ''),
                        'num_conformers': mol_3d.GetNumConformers()
                    })
                else:
                    self.stats['3d_embedding_failed'] += 1
                    self.failed_molecules['3d_embedding_failed'].append({
                        'name': name,
                        'smiles': row['smiles'],
                        'reason': '3D embedding failed - could not generate conformer'
                    })
                    
            except Exception as e:
                logger.debug(f"3D embedding failed for {name}: {e}")
                self.stats['3d_embedding_failed'] += 1
                self.failed_molecules['3d_embedding_failed'].append({
                    'name': name,
                    'smiles': row['smiles'],
                    'reason': f'3D embedding error: {str(e)}'
                })
                continue
        
        result_df = pd.DataFrame(results)
        self.stats['final_molecules'] = len(result_df)
        
        logger.info(f"Successfully generated 3D conformers for {len(result_df)} molecules")
        return result_df
    
    def _embed_molecule(self, mol: Chem.Mol, num_conformers: int, random_seed: int):
        """Embed molecule in 3D space."""
        try:
            # Use ETKDG algorithm with improved parameters
            params = AllChem.ETKDGv3()
            params.randomSeed = random_seed
            params.numThreads = 0  # Use all available cores
            params.useExpTorsionAnglePrefs = True
            params.useBasicKnowledge = True
            
            if num_conformers == 1:
                result = AllChem.EmbedMolecule(mol, params)
                return result != -1
            else:
                confIds = AllChem.EmbedMultipleConfs(mol, numConfs=num_conformers, params=params)
                return len(confIds) > 0
                
        except Exception:
            return False
    
    def _optimize_conformers(self, mol: Chem.Mol):
        """Optimize conformers using MMFF94."""
        try:
            if AllChem.MMFFHasAllMoleculeParams(mol):
                for confId in range(mol.GetNumConformers()):
                    AllChem.MMFFOptimizeMolecule(mol, confId=confId)
            else:
                # Fallback to UFF if MMFF94 parameters not available
                for confId in range(mol.GetNumConformers()):
                    AllChem.UFFOptimizeMolecule(mol, confId=confId)
        except Exception:
            pass  # If optimization fails, continue with unoptimized structure

    def save_failed_molecules(self, output_dir: str = "failed_molecules"):
        """
        Save failed molecules to separate files for debugging.
        
        Args:
            output_dir: Directory to save failed molecule files
        """
        if not any(self.failed_molecules.values()):
            logger.info("No failed molecules to save")
            return
        
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        total_failed = 0
        
        for failure_type, failed_mols in self.failed_molecules.items():
            if not failed_mols:
                continue
                
            output_file = Path(output_dir) / f"{failure_type}.tsv"
            
            # Save to TSV file with tab separation
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("Name\tSMILES\tFailure_Reason\n")
                for mol_info in failed_mols:
                    f.write(f"{mol_info['name']}\t{mol_info['smiles']}\t{mol_info['reason']}\n")
            
            logger.info(f"Saved {len(failed_mols)} {failure_type.replace('_', ' ')} to {output_file}")
            total_failed += len(failed_mols)
        
        # Create a summary file
        summary_file = Path(output_dir) / "failure_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("Failed Molecules Summary\n")
            f.write("=" * 50 + "\n\n")
            
            for failure_type, failed_mols in self.failed_molecules.items():
                if failed_mols:
                    f.write(f"{failure_type.replace('_', ' ').title()}: {len(failed_mols)} molecules\n")
            
            f.write(f"\nTotal failed molecules: {total_failed}\n")
            f.write(f"Success rate: {((self.stats['input_molecules'] - total_failed) / self.stats['input_molecules'] * 100):.1f}%\n")
        
        logger.info(f"Failure summary saved to {summary_file}")
        logger.info(f"All failed molecules saved to directory: {output_dir}")
    
    def get_failed_molecules_summary(self) -> Dict[str, int]:
        """Get a summary of failed molecules by failure type."""
        return {failure_type: len(failed_mols) 
                for failure_type, failed_mols in self.failed_molecules.items() 
                if failed_mols}

    
    def save_results(self, df: pd.DataFrame, output_file: str, include_props: bool = True):
        """Save results to SDF file."""
        logger.info(f"Saving {len(df)} molecules to {output_file}")
        
        try:
            properties = ['smiles', 'original_smiles'] if include_props else None
            PandasTools.WriteSDF(df, output_file, idName='name', molColName='mol', properties=properties)
            logger.info(f"Results saved to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            raise



    def print_statistics(self):
        """Print processing statistics."""
        logger.info("Processing Statistics:")
        logger.info("-" * 50)
        logger.info(f"Input molecules: {self.stats['input_molecules']}")
        logger.info(f"Duplicates removed: {self.stats['duplicates_removed']}")
        logger.info(f"Invalid SMILES: {self.stats['invalid_smiles']}")
        logger.info(f"Invalid atoms: {self.stats['invalid_atoms']}")
        logger.info(f"Lipinski violations: {self.stats['lipinski_violations']}")
        logger.info(f"Salt removal failed: {self.stats['salt_removal_failed']}")
        logger.info(f"Standardization failed: {self.stats['standardization_failed']}")
        logger.info(f"3D embedding failed: {self.stats['3d_embedding_failed']}")
        logger.info(f"Final molecules: {self.stats['final_molecules']}")
        logger.info("-" * 50)
        
        if self.stats['input_molecules'] > 0:
            success_rate = (self.stats['final_molecules'] / (self.stats['input_molecules']- self.stats['duplicates_removed'])) * 100
            logger.info(f"Overall success rate: {success_rate:.1f}%")


def main():
    parser = argparse.ArgumentParser(description='Convert SMILES to 3D SDF for virtual screening')
    parser.add_argument('input_file', help='Input file with SMILES')
    parser.add_argument('-o', '--output', help='Output SDF file (auto-generated if not provided)')
    parser.add_argument('-p', '--protonate', action='store_true', help='Apply pH-dependent protonation')
    parser.add_argument('--ph', type=float, default=7.4, help='Target pH for protonation (default: 7.4)')
    parser.add_argument('-l', '--lipinski', action='store_true', help='Apply Lipinski rule of five filtering')
    parser.add_argument('-c', '--conformers', type=int, default=1, help='Number of conformers to generate')
    parser.add_argument('--no-optimize', action='store_true', help='Skip conformer optimization')
    parser.add_argument('--permissive-atoms', action='store_true', help='Allow non-standard atoms')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--save-failed', type=str, default='failed_molecules', 
                       help='Directory to save failed molecules (default: failed_molecules)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Generate output filename if not provided
    if args.output is None:
        stem = Path(args.input_file).stem
        suffix = "_3d_protonated.sdf" if args.protonate else "_3d_standardized.sdf"
        args.output = f"{stem}{suffix}"
    
    # Initialize processor
    processor = MoleculeProcessor(
        apply_lipinski=args.lipinski,
        strict_atom_filter=not args.permissive_atoms
    )
    
    try:
        # Process molecules
        df = processor.read_smiles_file(args.input_file)
        df = processor.standardize_molecules(df, args.protonate, args.ph)
        
        if len(df) == 0:
            logger.error("No molecules survived standardization. Exiting.")
            return 1
        
        df = processor.generate_3d_conformers(
            df, 
            num_conformers=args.conformers,
            optimize=not args.no_optimize,
            random_seed=args.seed
        )
        
        if len(df) == 0:
            logger.error("No molecules survived 3D embedding. Exiting.")
            return 1
        
        processor.save_results(df, args.output)
        processor.print_statistics()
        
        # Save failed molecules for debugging
        processor.save_failed_molecules(args.save_failed)
        
        return 0
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
