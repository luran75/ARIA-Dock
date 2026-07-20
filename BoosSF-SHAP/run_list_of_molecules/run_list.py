import argparse
import os
from os import listdir
import datetime
import subprocess
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from pathlib import Path
from tqdm import tqdm
RDLogger.DisableLog('rdApp.*')


home_dir = os.getcwd()

parser = argparse.ArgumentParser(description = "Run BoostSF-SHAP on a list of receptor, reference, and ligand files.")

parser.add_argument("--receptor",  help="The path of the receptor file, pdb file format.")
parser.add_argument("--reference",    help="The path of the refrence file to carry out smina docking, pdb file format.")
parser.add_argument("--list",    help="The path of the sdf file containing the ligands to run.")
args = parser.parse_args()

receptor = args.receptor
reference = args.reference
list = args.list


# convert the protein and the reference to pdbqt format
subprocess.run(['obabel', '-ipdb', f'{receptor}', '-opdbqt', '-O', f'{Path(receptor).stem}.pdbqt', '-h', '--partialcharge', 'gasteiger'],stderr = subprocess.DEVNULL, stdout = subprocess.DEVNULL)
subprocess.run(['obabel', '-ipdb', f'{reference}', '-opdbqt', '-O', f'{Path(reference).stem}.pdbqt', '-h', '--partialcharge', 'gasteiger'],stderr = subprocess.DEVNULL, stdout = subprocess.DEVNULL)

ligands_dir = f'ligands_pdbqt_{Path(list).stem}'

os.makedirs(ligands_dir, exist_ok=True)

shutil.copy(list, f'{ligands_dir}/{list}')
os.chdir(ligands_dir)



# convert the list of ligands to pdbqt format

print(f"Converting ligands to PDBQT format...")
suppl = Chem.SDMolSupplier(list)
lig_filenames = []
failed_ligands = []
for idx, mol in tqdm(enumerate(suppl), total=len(suppl), desc="Converting ligands"):
    if mol is not None:
        # convert the ligand to pdbqt format
        if mol.HasProp('Molecule Name'):
            name = mol.GetProp('Molecule Name')

        elif mol.HasProp('_Name'):
            name = mol.GetProp('_Name')

        else:
            name = f'ligand_{idx}'

        pdbqt_lig = f'{name}.pdbqt'
        sdf_tmp = f'{name}.sdf'
        # first write the ligand to a temporary sdf file
        with Chem.SDWriter(sdf_tmp) as w:
            w.write(mol)

        # convert sdf to pdbqt using obabel
        result = subprocess.run(
            ['obabel', '-isdf', sdf_tmp, '-opdbqt', '-O', pdbqt_lig,
             '-h', '--partialcharge', 'gasteiger'],
            stderr=subprocess.PIPE, stdout=subprocess.PIPE
        )
        os.remove(sdf_tmp)

        ok = os.path.exists(pdbqt_lig) and os.path.getsize(pdbqt_lig) > 0
        if ok:
            with open(pdbqt_lig) as f:
                content = f.read()
            ok = 'ROOT' in content and 'TORSDOF' in content

        if ok:
            lig_filenames.append(pdbqt_lig)
        else:
            failed_ligands.append(name)
            print(f"WARNING: failed to convert {name} to a valid PDBQT")
            print(result.stderr.decode(errors='ignore'))

if failed_ligands:
    print(f"\n{len(failed_ligands)} ligand(s) failed conversion and were skipped: {failed_ligands}\n")


with open(f'{home_dir}/ligand_filenames.txt', 'w') as f:
    for lig in lig_filenames:
        f.write(f'{Path(receptor).stem}.pdbqt; {lig}; {Path(reference).stem}.pdbqt\n')


#shutil.move(f'{ligands_dir}/{list}', home_dir)
#os.chdir(home_dir)
#os.chdir(results_dir)

shutil.move(f'{home_dir}/{Path(receptor).stem}.pdbqt', f'{home_dir}/{ligands_dir}/{Path(receptor).stem}.pdbqt')
shutil.move(f'{home_dir}/{Path(reference).stem}.pdbqt', f'{home_dir}/{ligands_dir}/{Path(reference).stem}.pdbqt')



# Run BoostSF-SHAP for each ligand
print('Running BoostSF-SHAP for each ligand...')
subprocess.run(['python', f'{home_dir}/BoostSF-SHAP-main.py', '--modeltype', 'catboost', '--model', f'{home_dir}/models/catboost-BV/catBoost-V10_BV.cbm',  '--rllist', f'{home_dir}/ligand_filenames.txt'])

# move the results to the home directory
os.chdir(home_dir)
results_dir = [f for f in listdir(ligands_dir) if f.startswith("result")]
print(f"Found result directories: {results_dir}")


if results_dir:
    for result in results_dir:
        shutil.move(f'{ligands_dir}/{result}', f'{home_dir}/{result}')
