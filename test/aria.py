### conda activate aria_37_universal

import time

# Start timer
start_time = time.time()

import random
import argparse
import os
import sys
import subprocess
import shutil
import glob
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import PandasTools
from typing import Optional
from pathlib import Path
import collections
import pandas as pd
import datamol as dm
import re, logging
import math
import concurrent.futures
dm.disable_rdkit_log()
random.seed(0)

# RDKit imports
from rdkit import Chem
from rdkit.Chem import PandasTools
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import RDLogger
# Suppress everything except errors:
RDLogger.DisableLog('rdApp.info')

try:
	import datamol as dm
	DATAMOL_AVAILABLE = True
except ImportError:
	DATAMOL_AVAILABLE = False
	print("Warning: datamol not available. Using RDKit-only standardization.")

try:
	from openbabel import openbabel as ob
	OPENBABEL_AVAILABLE = True
	ob.obErrorLog.SetOutputLevel(0)
except ImportError:
	OPENBABEL_AVAILABLE = False
	print("Warning: OpenBabel not available. pH adjustment disabled.")
	
VALID_ATOMS = {'C', 'H', 'O', 'N', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'B', 'Si'}


# Get the current directory
home_directory = os.getcwd()


parser = argparse.ArgumentParser(prog='aria-dock',
								 description='Active learning platform for structure-based drug design',
								 epilog='Villoutreix group')
parser.add_argument('-p', '--protein', type=str, action='store', required=True, help='Target protein file in pdb format')
parser.add_argument('-l', '--ligand', type=str ,action='store', required=True, help='Reference ligand/fragments file in pdb format')
parser.add_argument('-d', '--database', type=str, action='store', required=True, help='Input database in .smi format')
parser.add_argument('-r', '--rounds', type=int, action='store', default=5, help='Number of rounds to perform')
parser.add_argument('-m', '--model', action='extend', nargs='*', type=str, choices=['auto', 'rf', 'ri', 'xgb', 'svr', 'knr'], default=None, help='Type of ML models to run. rf: RandomForest, ri: Ridge, xgb: XGBoost, svr: Support Vector Regression, knr: K-Nearest Regressor, auto: automatic selection based on metrics. Combination of multiple models is allowed: -e.g., -m rf ri xb')
parser.add_argument('-n', '--number', type=int, action='store', default=0, help='If auto is selected, then specify the number of models to use (from 1 to 5).')
parser.add_argument('--version', action='version', version='%(prog)s 1.0')
parser.add_argument('--log', action='store_true', default=False, help='Enables logging of the workflow progress and results')
parser.add_argument('--cpu', type=int, action='store', default=8, help='Number CPUs to expoit dursing the molecular docking step. Default: 8')
parser.add_argument('--docking_chunks', type=int, action='store', default=4, help='Number of chunks to split the docking job into for parallelization. Default: 4')
parser.add_argument('--percentage', type=int, action='store', default=1, help='Percentage of molecules to select for the first docking round. Default: 1')
parser.add_argument('--plug-in', action='extend', nargs="*", type=str, choices=['boostsf-shap'], default=['None'], help='Specify a plug-in to use with Ariadne: BoostSF-SHAP for post-processing analysis')
parser.add_argument('-v', '--verbose', action='store_true', default=False, help='Enable verbose output with detailed progress information')



# Parse command line arguments
args = parser.parse_args()

full_db = args.database
target_protein_file_input = args.protein
target_ligand_file = args.ligand
rounds = args.rounds
model = args.model if args.model is not None else 'auto'
if model == ['auto']:
	model = 'auto'
number_of_models = args.number
cpu = args.cpu
docking_chunks = args.docking_chunks
percentage = float((args.percentage)/100)
plug_in = args.plug_in
verbose = args.verbose

# Helper function for verbose printing
def vprint(*args, **kwargs):
	"""Print only if verbose mode is enabled"""
	if verbose:
		print(*args, **kwargs)

# define the directories to sort the files
mol_select_directory = f'{home_directory}/00_molecule_selection'
ml_directory = f'{home_directory}/01_ML'
docking_directory = f'{home_directory}/02_docking'
data_analysis_directory = f'{home_directory}/03_data_analysis'
post_processing_directory = f'{home_directory}/04_post-processing'
plugins_directory = f'{post_processing_directory}/plug-ins'
boostsfshap_directory = f'{plugins_directory}/BoostSF-SHAP'


if 'auto' in model and (number_of_models < 1 or number_of_models > 5):
	print('Please specify a valid number of models to use when auto is selected (from 1 to 5).')
	sys.exit(1)
if 'auto' in model and (('rf' or 'ri' or 'xb' or 'svr' or 'knr') in model):
		print(f'Please either select specific models to run or auto with the number of models to use. You entered {model}')
		sys.exit(1)

# logging setup

class LoggerWriter:
	def __init__(self, level):
		self.level = level
		self.buffer = ""
		self.last_tqdm_line = None

	def write(self, message):
		if message != '\n':
			self.buffer += message
		
		if '\n' in message:
			lines = (self.buffer).split('\n')
			for i, line in enumerate(lines[:-1]):  # Process all but the last (empty) line
				line = line.strip()
				if line:
					if self._is_tqdm_line(line):
						# Store the latest tqdm line, don't log it yet
						self.last_tqdm_line = line
					else:
						# Regular line - log it
						if self.last_tqdm_line and not self._is_tqdm_line(line):
							# Check if this looks like a completion (100% or similar)
							if "100%" in self.last_tqdm_line or "it/s" in self.last_tqdm_line:
								self.level("Final Progress: " + self.last_tqdm_line)
							self.last_tqdm_line = None
						self.level(line)
			
			self.buffer = lines[-1]  # Keep any remaining partial line

# for logging purposes
	def _is_tqdm_line(self, line):
		"""Check if a line looks like a tqdm progress bar"""
		tqdm_indicators = [
			'%|',  # percentage with progress bar
			'it/s',  # iterations per second
			's/it',  # seconds per iteration
			'\r',   # carriage return (though this might be stripped)
		]
		return any(indicator in line for indicator in tqdm_indicators)

	def flush(self):
		if self.buffer:
			line = self.buffer.strip()
			if line:
				if self._is_tqdm_line(line):
					# If it's a tqdm line and looks complete, log it
					if "100%" in line:
						self.level("Final Progress: " + line)
				else:
					self.level(line)
			self.buffer = ""
		
		# Also flush any stored tqdm line on explicit flush
		if self.last_tqdm_line:
			if "100%" in self.last_tqdm_line:
				self.level("Final Progress: " + self.last_tqdm_line)
			self.last_tqdm_line = None


plt.set_loglevel('WARNING') # to avoid matplotlib warnings

def logged_run(*popenargs, **kwargs):
	kwargs["capture_output"] = True
	kwargs["text"] = True

	result = _original_run(*popenargs, **kwargs)

	# tqdm handling for subprocess output
	tqdm_line = None
	if result.stdout:
		for line in result.stdout.splitlines():
			line = line.strip()
			if any(indicator in line for indicator in ['%|', 'it/s', 's/it']):
				tqdm_line = line  # keep updating with latest progress line
			else:
				logging.info(line)

	if result.stderr:
		for line in result.stderr.splitlines():
			line = line.strip()
			if any(indicator in line for indicator in ['%|', 'it/s', 's/it']):
				tqdm_line = line
			else:
				logging.error(line)

	# log final progress bar if one was captured and looks complete
	if tqdm_line and "100%" in tqdm_line:
		logging.info("Final Progress: " + tqdm_line)

	return result

# logging
if args.log:
	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s - %(levelname)s - %(message)s",
		handlers=[
			logging.FileHandler("aria_log.log", mode="w"),
			logging.StreamHandler(sys.__stdout__)]
	)
	sys.stdout = LoggerWriter(logging.info)
	sys.stderr = LoggerWriter(logging.error)
	_original_run = subprocess.run
	subprocess.run = logged_run

# functions

def sdf_to_csv(sdf_file, delimiter=','):
	"""
	Convert an SDF file to a CSV or TSV file.
	
	Args:
		sdf_file (str): Path to the input SDF file.
		output_file (str): Path to the output CSV or TSV file.
		delimiter (str, optional): Delimiter to use in the output file. Default is ',' (CSV).
	"""
	# Read the SDF file
	suppl = Chem.SDMolSupplier(sdf_file)

	# Convert to a pandas DataFrame
	df = PandasTools.LoadSDF(sdf_file, smilesName='SMILES', molColName='Molecule', includeFingerprints=False)

	# Drop the 'Molecule' column if present
	if 'Molecule' in df.columns:
		df = df.drop('Molecule', axis=1)
		
	df['minimizedAffinity'] = df['minimizedAffinity'].astype(float)
	df = df[df['minimizedAffinity'] < 0] # if >0, then smina artifact
	df.reset_index(drop = True) 

	# Save the DataFrame as CSV or TSV
	output_file = f'{Path(sdf_file).stem}.csv'
	df.to_csv(output_file, sep=delimiter, index=False, columns = ['SMILES', 'ID', 'minimizedAffinity'])
	
	return output_file    

def is_valid_smiles_format(smiles: str) -> bool:
	"""Basic SMILES format validation."""
	if not smiles or len(smiles) < 2:
		return False
	# Skip obvious metal complexes with many dots
	if smiles.count('.') > 1:
		return False
	return True

def read_smiles_file(filename: str) -> pd.DataFrame:
	"""
	Read SMILES from file with improved error handling.
		
	Args:
		filename: Path to input file
			
	Returns:
		DataFrame with 'smiles' and 'name' columns
	"""
	print(f"Reading SMILES from {filename}")
		
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
				if is_valid_smiles_format(smiles):
					rows.append({'smiles': smiles, 'name': name, 'line_number': line_num})
				#else:
					#print(f"Invalid SMILES format at line {line_num}: {smiles}")
			elif len(parts) == 1:
				# No name provided, use line number
				smiles = parts[0]
				if is_valid_smiles_format(smiles):
					rows.append({'smiles': smiles, 'name': f"mol_{line_num}", 'line_number': line_num})
		
	if not rows:
		raise ValueError("No valid SMILES found in input file")
		
	df = pd.DataFrame(rows)
	df.drop(columns= 'line_number', inplace=True)
	
	# Remove duplicates
	initial_count = len(df)
	df = df.drop_duplicates(subset='smiles', keep='first').reset_index(drop=True)
	duplicates_removed = initial_count - len(df)
	print(f"Loaded {len(df)} unique molecules ({duplicates_removed} duplicates removed)")
	return df



def remove_salts(mol: Chem.Mol) -> Optional[Chem.Mol]:
	"""Remove salts from molecule."""
	salt_remover = SaltRemover()
	try:
		return salt_remover.StripMol(mol)
	except:
		return None
	
def has_valid_atoms(mol: Chem.Mol) -> bool:
	"""Check if molecule contains only valid atoms."""
	if mol is None:
		return False
		
	for atom in mol.GetAtoms():
		if atom.GetSymbol() not in VALID_ATOMS:
			return False
	return True

	
def standardize_molecule(mol: Chem.Mol):
	"""
	Apply molecular standardization.
	"""

	standardizer = rdMolStandardize.Normalizer()
	try:
		# Use datamol if available
		if DATAMOL_AVAILABLE:
			mol = dm.fix_mol(mol)
			if mol is None:
				return None
			mol = dm.sanitize_mol(mol, sanifix=True, charge_neutral=True)
			if mol is None:
				return None
			mol = dm.standardize_mol(mol, disconnect_metals=True, normalize=True, reionize=True, uncharge=True, stereo=True)
		else:
			# RDKit-only standardization
			mol = standardizer.normalize(mol)
			Chem.SanitizeMol(mol)
			mol = rdMolStandardize.Uncharger().uncharge(mol)
			
		return mol
			
	except Exception as e:
		print(f"Standardization failed: {e}")
		return None


def standardize_molecules(df: pd.DataFrame) -> pd.DataFrame:
	"""
	Standardize SMILES with multiple approaches.
		
	Args:
		df: DataFrame with SMILES
	Returns:
		DataFrame with standardized molecules
	"""
	print("Standardizing molecules...")
		
	standardization_failed = 0
	invalid_smiles = 0
	salt_removal_failed = 0
	invalid_atoms = 0
	failed_molecules = {
		'invalid_smiles': [],
		'invalid_atoms': [],
		'salt_removal_failed': [],
		'standardization_failed': [],
	}
	
	results = []

		
	for idx, row in tqdm(df.iterrows(), total=len(df), desc="Standardizing"):
		smiles = row['smiles']
		name = row['name']
		
		try:
			# Convert to molecule
			mol = Chem.MolFromSmiles(smiles)
			if mol is None:
				invalid_smiles += 1
				failed_molecules['invalid_smiles'].append({
					'name': name,
					'smiles': smiles,
					'reason': 'Could not parse SMILES string'
				})
				continue
				
			# Remove salts
			mol = remove_salts(mol)
			if mol is None:
				salt_removal_failed += 1
				failed_molecules['salt_removal_failed'].append({
					'name': name,
					'smiles': smiles,
					'reason': 'Salt removal failed'
				})
				continue
				
			# Atom filtering
			if not has_valid_atoms(mol):
				invalid_atoms += 1
				invalid_atoms_type = [atom.GetSymbol() for atom in mol.GetAtoms() 
							   if atom.GetSymbol() not in VALID_ATOMS]
				failed_molecules['invalid_atoms'].append({
					'name': name,
					'smiles': smiles,
					'reason': f'Invalid atoms detected: {list(set(invalid_atoms_type))}'
				})
				continue
				
			# Standardization
			mol = standardize_molecule(mol)
			if mol is None:
				standardization_failed += 1
				failed_molecules['standardization_failed'].append({
					'name': name,
					'smiles': smiles,
					'reason': 'Molecule standardization failed'
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
			print(f"Error processing {name}: {e}")
			standardization_failed += 1
			failed_molecules['standardization_failed'].append({
				'name': name,
				'smiles': smiles,
				'reason': f'Processing error: {str(e)}'
			})
			continue
		
	result_df = pd.DataFrame(results)
	print(f"Standardized {len(result_df)} molecules")

	vprint(failed_molecules)
	return result_df


def prep_database_for_ML_training(docking_file, full_database):
	'''
	Generates the input files for ML training in a memory-efficient way.
	- Creates a file for the ML training/test set from the docked molecules.
	- Creates a file with the remaining molecules from the full database that have not been docked.
	'''
	print("Preparing database for ML training...")
	
	# Read the docked molecules
	dock_df = pd.read_csv(docking_file, sep=',', names=['SMILES', 'ID', 'minimizedAffinity'], header=0, dtype={'SMILES': str, 'ID': str, 'minimizedAffinity': float})
	
	# The training set is just the docked molecules, formatted for the ML script (SMILES, Name, Score)
	output_file_train_test = f'{Path(docking_file).stem}_ML_test_train_set.smi'
	dock_df.to_csv(output_file_train_test, index=False, header=False, sep=' ')
	vprint(f"Created training/test set: {output_file_train_test}")

	# 2. Create a set of names of the docked molecules for fast lookup.
	docked_names = set(dock_df['ID'])
	
	# 3. Stream the full database to find remaining molecules without loading it all into memory.
	output_file_remaining = f'{Path(docking_file).stem}_MLtraining_remaining_mols.smi'
	
	vprint(f"Screening full database to find remaining molecules...")
	count_remaining = 0
	with open(full_database, 'r') as f_in, open(output_file_remaining, 'w') as f_out:
		for line in f_in:
			parts = line.strip().split()
			if len(parts) >= 2: # Ensure the line has at least SMILES and a name
				name = parts[1]
				if name not in docked_names:
					f_out.write(line)
					count_remaining += 1
	
	vprint(f"Found {count_remaining} remaining molecules.")
	vprint(f"Created remaining molecules file: {output_file_remaining}")
	
	return output_file_train_test, output_file_remaining
	

def get_length(file):
	"""
	Obtain the number of molecules in an CSV or SDF file without converting it into a df (memory save).
	
	Args:
		file (str): Path to the input SDF or CSV file.
	"""
	m = 0
	  
	if file.endswith('.sdf'):
		with open(file, 'r') as f:
			for line in f:
				if line.startswith('$$$$'):
						m += 1
	
	elif file.endswith('.csv'):
		with open(file) as f:
			m = int(sum(1 for line in f))
				  
	else:
		print('File format not recognized, please provide an SDF or CSV file')
		m = 0

	return m



def clean_directory(src, dest):
	'''
	Move all the files that are not python scripts to a directory, including subdirectories
	'''
	home_dir = os.getcwd()
	os.chdir(f'{src}')
	files_to_move = [file for file in glob.glob("*") if not file.endswith(".py")] # move everything except the python scripts
	for f in files_to_move:
		shutil.move(f'{src}/{f}', f'{dest}/{f}')
	# return to the home directory
	os.chdir(home_dir)

def clean_directory_no_dir(src, dest):
	'''
	Move all the files that are not python scripts to a directory, excluding subdirectories
	'''
	home_dir = os.getcwd()
	os.chdir(f'{src}')
	all_items = glob.glob("*")
	# Only move files that don't end with .py (skip all directories)
	files_to_move = []
	for item in all_items:
		if os.path.isfile(item) and not item.endswith(".py"):
			files_to_move.append(item)
	for file in files_to_move:
		shutil.move(f'{src}/{file}', f'{dest}/{file}')
	os.chdir(home_dir)

# for consensus scoring
def calculate_consensus_score(input_files, strategy='rank_by_rank'):
	"""
	Calculates a consensus score from multiple prediction files in a more memory-efficient way.
	It takes a list of files as input.
	"""
	if not input_files:
		return pd.DataFrame()

	# Get model names from file names - internal naming convention
	model_names = [Path(f).stem.split('_')[0] for f in input_files]

	# Read the first file and prepare it as the base dataframe
	base_model_name = model_names[0]
	base_score_col = f'{base_model_name}_pred_score'
	base_df = pd.read_csv(input_files[0], header=0, dtype={'smiles': str, 'name': str, 'PredictedScore': float})
	base_df.columns = ['smiles', 'name', base_score_col]
	
	# Drop positive scores and missing values as they are likely artifacts
	base_df = base_df.dropna(subset=[base_score_col])
	base_df = base_df[base_df[base_score_col] < 0].reset_index(drop=True)

	# Iteratively read and merge the score column from other prediction files
	for i in range(1, len(input_files)):
		model_name = model_names[i]
		score_col = f'{model_name}_pred_score'
		
		# Read only necessary columns to save memory.
		temp_df = pd.read_csv(input_files[i], header=0, names=['smiles', 'name', 'PredictedScore'], usecols=['name', 'PredictedScore'], dtype={'name': str, 'PredictedScore': float})
		temp_df.columns = ['name', score_col]

		# Drop positive scores
		temp_df = temp_df[temp_df[score_col] < 0].reset_index(drop=True)

		# Merge the new score column into the base dataframe
		base_df = pd.merge(base_df, temp_df, on='name', how='inner')

	# An inner merge should prevent duplicates if 'name' is unique in each file, but we drop just in case.
	base_df.drop_duplicates('name', inplace=True)
	
	if strategy == 'rank_by_rank':
		score_columns = [f'{name}_pred_score' for name in model_names]
		
		# Compute rankings (lower scores = better ranking)
		base_df[score_columns] = base_df[score_columns].rank(axis=0, method="average", ascending=True)
		# Compute the mean rank across all scoring functions
		base_df["mean_rank"] = base_df[score_columns].mean(axis=1)
		
		# Sort molecules by Mean Rank (lower is better)
		base_df = base_df.sort_values(by="mean_rank", ignore_index=True)
		
		return base_df
	
	# Implementation for other strategies ('mean', 'min', 'max', 'median') could be added here if needed.
	
	return base_df


# for parallelization of chunks processing
def split_smi_file(input_file, num_chunks=4):
	"""Split a SMILES file into specified number of chunks."""
	
	# Count total molecules in the SMILES file
	with open(input_file, 'r') as f:
		lines = f.readlines()
		# Remove empty lines and comments
		molecules = [line.strip() for line in lines if line.strip() and not line.startswith('#')]
	
	total_molecules = len(molecules)
	chunk_size = math.ceil(total_molecules / num_chunks)
	if chunk_size > 10000: # set the upper limit of chunk dimension to 10000
		chunk_size = 10000
		num_chunks = int(math.ceil(total_molecules / chunk_size))
		
	print(f'\nSplitting {input_file} into {num_chunks} chunks...')
	print(f'Total molecules: {total_molecules}')
	print(f'Molecules per chunk: {chunk_size}')
	
	chunk_files = []
	input_stem = Path(input_file).stem
	
	for i in range(num_chunks):
		start_idx = i * chunk_size
		end_idx = min((i + 1) * chunk_size, total_molecules)
		
		if start_idx >= total_molecules:
			break
			
		chunk_filename = f'{input_stem}_chunk_{i+1}.smi'
		chunk_files.append(chunk_filename)
		
		with open(chunk_filename, 'w') as f:
			for j in range(start_idx, end_idx):
				f.write(molecules[j] + '\n')
		
		vprint(f'Created {chunk_filename} with {end_idx - start_idx} molecules')
	
	return chunk_files

def process_chunk(chunk_file, target_protein_file, target_ligand_file, strategy, r, cpu):
	'''
	Process a single chunk: 3D conversion -> protonation -> cleaning -> docking
	'''
	try:
		current_dir = os.getcwd()
		shutil.copy(f'{docking_directory}/preprocess.py', current_dir)
		chunk_stem = Path(chunk_file).stem
		vprint(f'\n=== Processing {chunk_file} ===')
		
		# 3D conversion and protonation
		vprint(f'Converting {chunk_file} to 3D...')
		result = subprocess.run([f'python preprocess.py {chunk_file} -o {chunk_stem}_3D_H.sdf -p --save-failed failed_molecules_{chunk_stem}'], 
							  shell=True, capture_output=True, text=True)
		if result.returncode != 0:
			raise Exception(f"3D conversion failed for {chunk_file}. Error: {result.stderr}")
		
		# Check if protonated file was created and is not empty
		if not os.path.exists(f'{chunk_stem}_3D_H.sdf') or os.path.getsize(f'{chunk_stem}_3D_H.sdf') == 0:
			raise Exception(f"3D conversion and protonation produced empty or missing file for {chunk_file}")
		
		# Clean nulls
		vprint(f'Cleaning {chunk_file}...')
		result = subprocess.run(["python", 'remove_null.py', f'{chunk_stem}_3D_H.sdf'], 
							  capture_output=True, text=True)
		if result.returncode != 0:
			raise Exception(f"Cleaning failed for {chunk_file}. Error: {result.stderr}")
		
		# Check if cleaned file exists
		if not os.path.exists(f'{chunk_stem}_3D_H_no_nulls.sdf'):
			raise Exception(f"Cleaning script did not produce expected output file for {chunk_file}")
		
		# Docking
		docking_output_file = f'{chunk_stem}_docked_exha16_1pose_{strategy}_r{r}.sdf'
		
		cmd = (f'python smina_progress.py --seed 0 '
			   f'-r {target_protein_file} '
			   f'-l {chunk_stem}_3D_H_no_nulls.sdf '
			   f'--autobox_ligand {target_ligand_file} '
			   f'--autobox_add 6 --exhaustiveness 16 --num_modes 1 '
			   f'-o {docking_output_file} '
			   f'--cpu {cpu}')
		
		vprint(f'Docking {chunk_file}...')
		result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
		if result.returncode != 0:
			raise Exception(f"Docking failed for {chunk_file}. Error: {result.stderr}")
		
		# Check if docking output was created
		if not os.path.exists(docking_output_file) or os.path.getsize(docking_output_file) == 0:
			raise Exception(f"Docking produced empty or missing output file for {chunk_file}")
		
		print(f'Successfully processed chunk {Path(chunk_file).stem[-1]}')
		return docking_output_file
		
	except Exception as e:
		print(f'ERROR: Failed to process {chunk_file}: {str(e)}')
		raise  # Re-raise the exception to stop the entire process


def merge_results(files, final_output_file):
	"""Merge all files into a single file, removing duplicate headers if present."""
	vprint(f'\nMerging files into {final_output_file}...')
	
	valid_files = [f for f in files if f is not None and os.path.exists(f)]
	if not valid_files:
		raise Exception('No valid files to merge!')
	
	# Step 1: Collect candidate headers (first line of each file)
	headers = []
	for f in valid_files:
		with open(f, 'r') as infile:
			first_line = infile.readline().strip()
			headers.append(first_line)
	
	# Step 2: Find a "common header" across files to make sure not to write it again and again
	common_header = None
	for h in set(headers):
		if headers.count(h) > 1:
			common_header = h
			break
	
	with open(final_output_file, 'w') as outfile:
		for i, docked_file in enumerate(valid_files):
			vprint(f'Merging {docked_file}...')
			with open(docked_file, 'r') as infile:
				lines = infile.readlines()
				
				if i == 0:
					# Always write full first file
					outfile.writelines(lines)
				else:
					if common_header and lines and lines[0].strip() == common_header:
						# Skip header if it matches the detected one
						outfile.writelines(lines[1:])
					else:
						outfile.writelines(lines)
				
				# Ensure newline separation
				if lines and not lines[-1].endswith('\n'):
					outfile.write('\n')
	
	vprint(f'Successfully merged {len(valid_files)} files into {final_output_file}')
	return True



# For machine learning
def run_rf_training(ml_directory, data_analysis_directory, train_test_file, remaining_mols_file, full_database, r, cpus_per_model):
	"""Runs the Random Forest training and prediction process."""
	rf_directory = f'{ml_directory}/RF'
	original_cwd = os.getcwd()
	os.chdir(rf_directory)

	try:
		shutil.copy(f'{ml_directory}/{train_test_file}', '.')
		shutil.copy(f'{ml_directory}/{remaining_mols_file}', '.')
	
		# Construct full path to database file
		full_database_path = os.path.join(ml_directory, os.path.basename(full_database))
		if not os.path.exists(os.path.basename(full_database)):
			shutil.copy(full_database_path, '.')

		# Train the ML algorithm
		if verbose:
			subprocess.run(["python", '01_RF_train.py', train_test_file, remaining_mols_file, str(r), '--cpu', str(cpus_per_model), '--verbose'], check=True)
		else:
			subprocess.run(["python", '01_RF_train.py', train_test_file, remaining_mols_file, str(r), '--cpu', str(cpus_per_model)], check=True)

		# Create chunks of the full database if it's too large
		chunk_files = split_smi_file(full_database)
		print('Predicting the scores on the full dataset...')
		prediction_script = '02_RF_apply.py'
		chunked_preds = []
		for i, chunk in enumerate(tqdm(chunk_files, desc = f'Predicting RF scores in chunks', total = len(chunk_files))):
			if os.path.exists(prediction_script):
				subprocess.run(["python", prediction_script, f'rf_model_and_info_r{r}.pkl', os.path.basename(chunk), str(r)], check=True)
				pred_chunk_file = f"RF_regression-grid-predicted-scores_c{i+1}_r{r}.csv"
				chunked_preds.append(pred_chunk_file)

		rf_pred_output_file = f"RF_regression-grid-predicted-scores_r{r}.csv"
		rf_metrics_file = f'RF_metrics_r{r}.csv'
		
		
		metrics_df = pd.read_csv(rf_metrics_file, sep=',', header = 0)
		r2_val = float(metrics_df['R2 val'])
		r2_train = float(metrics_df['R2 train'])
		
		merge_results(chunked_preds, rf_pred_output_file)
		for f in chunk_files: # remove the chunk files once used
			os.remove(f)
		for f in chunked_preds: # remove the chunk predictions once used
			os.remove(f)		

		return rf_pred_output_file, rf_metrics_file, r2_train, r2_val
	finally:
		os.chdir(original_cwd)

def run_ridge_training(ml_directory, data_analysis_directory, train_test_file, remaining_mols_file, full_database, r, cpus_per_model):
	"""Runs the Ridge regression training and prediction process."""
	ridge_directory = f'{ml_directory}/ridge'
	original_cwd = os.getcwd()
	os.chdir(ridge_directory)

	try:
		shutil.copy(f'{ml_directory}/{train_test_file}', '.')
		shutil.copy(f'{ml_directory}/{remaining_mols_file}', '.')
	
		# Construct full path to database file
		full_database_path = os.path.join(ml_directory, os.path.basename(full_database))
		if not os.path.exists(os.path.basename(full_database)):
			shutil.copy(full_database_path, '.')

		# Train the ML algorithm
		if verbose:
			subprocess.run(["python", '01_ridge_train.py', train_test_file, remaining_mols_file, str(r), '--cpu', str(cpus_per_model), '--verbose'], check=True)
		else:
			subprocess.run(["python", '01_ridge_train.py', train_test_file, remaining_mols_file, str(r), '--cpu', str(cpus_per_model)], check=True)
		
		# Use the algorithm on the whole library
		chunk_files = split_smi_file(full_database)
		print('Predicting the scores on the full dataset...')
		prediction_script = '02_ridge_apply.py'
		chunked_preds = []
		for i, chunk in enumerate(tqdm(chunk_files, desc = f'Predicting Ridge scores in chunks', total = len(chunk_files))):
			if os.path.exists(prediction_script):
				subprocess.run(["python", prediction_script, f'ridge_model_and_info_r{r}.pkl', os.path.basename(chunk), str(r)], check=True)
				pred_chunk_file = f"Ridge_regression-grid-predicted-scores_c{i+1}_r{r}.csv"
				chunked_preds.append(pred_chunk_file)
		
		ridge_pred_output_file = f'Ridge_regression-optuna-predicted-scores_r{r}.csv'
		ridge_metrics_file = f'Ridge_metrics_r{r}.csv'

		metrics_df = pd.read_csv(ridge_metrics_file, sep=',', header = 0)
		r2_val = float(metrics_df['R2 val'])
		r2_train = float(metrics_df['R2 train'])

		merge_results(chunked_preds, ridge_pred_output_file)
		for f in chunk_files: # remove the chunk files once used
			os.remove(f)
		for f in chunked_preds: # remove the chunk predictions once used
			os.remove(f)	
		return ridge_pred_output_file, ridge_metrics_file, r2_train, r2_val
	finally:
		os.chdir(original_cwd)

def run_xgb_training(ml_directory, data_analysis_directory, train_test_file, remaining_mols_file, full_database, r, cpus_per_model):
	"""Runs the XGBoost training and prediction process."""
	xgb_directory = f'{ml_directory}/xgboost'
	original_cwd = os.getcwd()
	os.chdir(xgb_directory)

	try:
		shutil.copy(f'{ml_directory}/{train_test_file}', '.')
		shutil.copy(f'{ml_directory}/{remaining_mols_file}', '.')
	
		# Construct full path to database file
		full_database_path = os.path.join(ml_directory, os.path.basename(full_database))
		if not os.path.exists(os.path.basename(full_database)):
			shutil.copy(full_database_path, '.')

		# Train the ML algorithm
		if verbose:
			subprocess.run(["python", '01_xgboost_train.py', train_test_file, remaining_mols_file, str(r), '--cpu', str(cpus_per_model), '--verbose'], check=True)
		else:
			subprocess.run(["python", '01_xgboost_train.py', train_test_file, remaining_mols_file, str(r), '--cpu', str(cpus_per_model)], check=True)

		# Use the algorithm on the whole library
		chunk_files = split_smi_file(full_database)
		print('Predicting the scores on the full dataset...')
		prediction_script = '02_xgboost_apply.py'
		chunked_preds = []
		for i, chunk in enumerate(tqdm(chunk_files, desc = f'Predicting XGBoost scores in chunks', total = len(chunk_files))):
			if os.path.exists(prediction_script):
				subprocess.run(["python", prediction_script, f'xgb_model_and_info_r{r}.pkl', os.path.basename(chunk), str(r)], check=True)
				pred_chunk_file = f"XGBoost_regression-grid-predicted-scores_c{i+1}_r{r}.csv"
				chunked_preds.append(pred_chunk_file)
		
		xgb_pred_output_file = f'XGBoost_regression-optuna-predicted-scores_r{r}.csv'
		xgb_metrics_file = f'XGBoost_metrics_r{r}.csv'

		metrics_df = pd.read_csv(xgb_metrics_file, sep=',', header = 0)
		r2_val = float(metrics_df['R2 val'])
		r2_train = float(metrics_df['R2 train'])

		merge_results(chunked_preds, xgb_pred_output_file)
		for f in chunk_files: # remove the chunk files once used
			os.remove(f)
		for f in chunked_preds: # remove the chunk predictions once used
			os.remove(f)	

		return xgb_pred_output_file, xgb_metrics_file, r2_train, r2_val
	finally:
		os.chdir(original_cwd)


def run_knr_training(ml_directory, data_analysis_directory, train_test_file, remaining_mols_file, full_database, r, cpus_per_model):
	"""Runs the Random Forest training and prediction process."""
	knr_directory = f'{ml_directory}/KNR'
	original_cwd = os.getcwd()
	os.chdir(knr_directory)

	try:
		shutil.copy(f'{ml_directory}/{train_test_file}', '.')
		shutil.copy(f'{ml_directory}/{remaining_mols_file}', '.')
	
		# Construct full path to database file
		full_database_path = os.path.join(ml_directory, os.path.basename(full_database))
		if not os.path.exists(os.path.basename(full_database)):
			shutil.copy(full_database_path, '.')

		# Train the ML algorithm
		if verbose:
			subprocess.run(["python", '01_KNR_train.py', train_test_file, remaining_mols_file, str(r), '--cpu', str(cpus_per_model), '--verbose'], check=True)
		else:
			subprocess.run(["python", '01_KNR_train.py', train_test_file, remaining_mols_file, str(r), '--cpu', str(cpus_per_model)], check=True)

		# Use the algorithm on the whole library

		# Create chunks of the full database if it's too large
		chunk_files = split_smi_file(full_database)
		print('Predicting the scores on the full dataset...')
		prediction_script = '02_KNR_apply.py'
		chunked_preds = []
		for i, chunk in enumerate(tqdm(chunk_files, desc = f'Predicting KNR scores in chunks', total = len(chunk_files))):
			if os.path.exists(prediction_script):
				subprocess.run(["python", prediction_script, f'KNR_model_and_info_r{r}.pkl', os.path.basename(chunk), str(r)], check=True)
				pred_chunk_file = f"KNR_regression-grid-predicted-scores_c{i+1}_r{r}.csv"
				chunked_preds.append(pred_chunk_file)

		knr_pred_output_file = f"KNR_regression-grid-predicted-scores_r{r}.csv"
		knr_metrics_file = f'KNR_metrics_r{r}.csv'

		metrics_df = pd.read_csv(knr_metrics_file, sep=',', header = 0)
		r2_val = float(metrics_df['R2 val'])
		r2_train = float(metrics_df['R2 train'])

		merge_results(chunked_preds, knr_pred_output_file)
		for f in chunk_files: # remove the chunk files once used
			os.remove(f)
		for f in chunked_preds: # remove the chunk predictions once used
			os.remove(f)	

		return knr_pred_output_file, knr_metrics_file, r2_train, r2_val
	finally:
		os.chdir(original_cwd)


def run_svr_training(ml_directory, data_analysis_directory, train_test_file, remaining_mols_file, full_database, r, cpus_per_model):
	"""Runs the Random Forest training and prediction process."""
	svr_directory = f'{ml_directory}/SVR'
	original_cwd = os.getcwd()
	os.chdir(svr_directory)

	try:
		shutil.copy(f'{ml_directory}/{train_test_file}', '.')
		shutil.copy(f'{ml_directory}/{remaining_mols_file}', '.')
	
		# Construct full path to database file
		full_database_path = os.path.join(ml_directory, os.path.basename(full_database))
		if not os.path.exists(os.path.basename(full_database)):
			shutil.copy(full_database_path, '.')

		# Train the ML algorithm
		if verbose:
			subprocess.run(["python", '01_SVR_train.py', train_test_file, remaining_mols_file, str(r), '--cpu', str(cpus_per_model), '--verbose'], check=True)
		else:
			subprocess.run(["python", '01_SVR_train.py', train_test_file, remaining_mols_file, str(r), '--cpu', str(cpus_per_model)], check=True)

		# Use the algorithm on the whole library

		# Create chunks of the full database if it's too large
		chunk_files = split_smi_file(full_database)
		print('Predicting the scores on the full dataset...')
		prediction_script = '02_SVR_apply.py'
		chunked_preds = []
		for i, chunk in enumerate(tqdm(chunk_files, desc = f'Predicting SVR scores in chunks', total = len(chunk_files))):
			if os.path.exists(prediction_script):
				subprocess.run(["python", prediction_script, f'SVR_model_and_info_r{r}.pkl', os.path.basename(chunk), str(r)], check=True)
				pred_chunk_file = f"SVR_regression-grid-predicted-scores_c{i+1}_r{r}.csv"
				chunked_preds.append(pred_chunk_file)

		svr_pred_output_file = f"SVR_regression-grid-predicted-scores_r{r}.csv"
		svr_metrics_file = f'SVR_metrics_r{r}.csv'

		metrics_df = pd.read_csv(svr_metrics_file, sep=',', header = 0)
		r2_val = float(metrics_df['R2 val'])
		r2_train = float(metrics_df['R2 train'])

		merge_results(chunked_preds, svr_pred_output_file)
		for f in chunk_files: # remove the chunk files once used
			os.remove(f)
		for f in chunked_preds: # remove the chunk predictions once used
			os.remove(f)	
					
		return svr_pred_output_file, svr_metrics_file, r2_train, r2_val
	finally:
		os.chdir(original_cwd)


def cleanup_intermediate_files(chunk_files):
	'''
	Clean up all intermediate files created during processing
	'''
	vprint('\nCleaning up intermediate files...')
	
	files_removed = 0
	for chunk_file in chunk_files:
		chunk_stem = Path(chunk_file).stem
		# Remove intermediate files for this chunk
		files_to_remove = [
			chunk_file,
			f'{chunk_stem}_3D.sdf',
			f'{chunk_stem}_3D_H.sdf',
			f'{chunk_stem}_3D_H_no_nulls.sdf'
		]
		
		for file_to_remove in files_to_remove:
			if os.path.exists(file_to_remove):
				try:
					os.remove(file_to_remove)
					files_removed += 1
					vprint(f'Removed {file_to_remove}')
				except Exception as e:
					print(f'Warning: Could not remove {file_to_remove}: {e}')
	
	vprint(f'Cleaned up {files_removed} intermediate files')

def parallel_docking_workflow(output_to_redock, target_protein_file, target_ligand_file, 
							cpu, strategy='start', r=0, num_chunks=4, cleanup_intermediate=True):
	'''
	Main function to run the parallel docking workflow
	'''
	
	output_to_redock_stem = Path(output_to_redock).stem
	chunk_files = []
	
	try:
		# Step 1: Split the input SMILES file into chunks
		chunk_files = split_smi_file(output_to_redock, num_chunks)
		
		if not chunk_files:
			raise Exception('No chunks created from input file')
		
		# Step 2: Process chunks in parallel
		print(f'\n=== Starting parallel processing of {len(chunk_files)} chunks for ROUND {r} ===')
		vprint(f'Using {cpu} CPUs per chunk')
		
		with concurrent.futures.ThreadPoolExecutor(max_workers=num_chunks) as executor:
			# Submit all chunk processing tasks
			future_to_chunk = {
				executor.submit(process_chunk, chunk_file, target_protein_file, 
							  target_ligand_file, strategy, r, cpu): chunk_file
				for chunk_file in chunk_files
			}
			
			# Collect results as they complete
			docked_files = []
			for future in concurrent.futures.as_completed(future_to_chunk):
				chunk_file = future_to_chunk[future]
				try:
					result = future.result()
					docked_files.append(result)
					vprint(f'Chunk {chunk_file} completed successfully')
				except Exception as exc:
					error_msg = f'CRITICAL ERROR: Chunk {chunk_file} failed: {exc}'
					print(error_msg)
					# Cancel remaining tasks
					for remaining_future in future_to_chunk:
						remaining_future.cancel()
					raise Exception(error_msg)
		
		# Step 3: Merge results
		final_output_file = f'all_mean_redocked_mols_exha16_1pose_{strategy}_r{r}.sdf'
		merge_results(docked_files, final_output_file)
		
		print(f'\n=== Parallel docking completed successfully! ===')
		print(f'Final output: {final_output_file}')
		
		# Step 4: Cleanup intermediate files
		if cleanup_intermediate:
			cleanup_intermediate_files(chunk_files)
		
		return final_output_file
		
	except Exception as e:
		print(f'\n=== PROCESS STOPPED DUE TO ERROR ===')
		print(f'Error: {str(e)}')
		
		# Cleanup chunks that were created before the error
		if cleanup_intermediate and chunk_files:
			vprint('\nCleaning up partial results...')
			cleanup_intermediate_files(chunk_files)
		
		raise 



# Create some directories to sort all of the data generated by the workflow
isExist = os.path.exists(data_analysis_directory)
if not isExist:
	os.makedirs(data_analysis_directory)

data_viz_directory = f'{data_analysis_directory}/data_viz'
isExist = os.path.exists(data_viz_directory)
if not isExist:
	os.makedirs(data_viz_directory)

sdf_directory = f'{home_directory}/05_sdf_files'
isExist = os.path.exists(sdf_directory)
if not isExist:
	os.makedirs(sdf_directory)

data_directory_name = '06_data'
isExist = os.path.exists(data_directory_name)
if not isExist:
	os.makedirs(data_directory_name)
data_directory = f'{home_directory}/{data_directory_name}'

ml_data_directory = f'{data_directory}/01_ML'
isExist = os.path.exists(ml_data_directory)
if not isExist:
	os.makedirs(ml_data_directory)

rf_data_directory = f'{ml_data_directory}/RF'
isExist = os.path.exists(rf_data_directory)
if not isExist:
	os.makedirs(rf_data_directory)

ridge_data_directory = f'{ml_data_directory}/Ridge'
isExist = os.path.exists(ridge_data_directory)
if not isExist:
	os.makedirs(ridge_data_directory)

xgboost_data_directory = f'{ml_data_directory}/XGBoost'
isExist = os.path.exists(xgboost_data_directory)
if not isExist:
	os.makedirs(xgboost_data_directory)

knr_data_directory = f'{ml_data_directory}/KNR'
isExist = os.path.exists(knr_data_directory)
if not isExist:
	os.makedirs(knr_data_directory)

svr_data_directory = f'{ml_data_directory}/SVR'
isExist = os.path.exists(svr_data_directory)
if not isExist:
	os.makedirs(svr_data_directory)

mol_select_data_directory = f'{data_directory}/00_Molecule_Selection'
isExist = os.path.exists(mol_select_data_directory)
if not isExist:
	os.makedirs(mol_select_data_directory)

docking_data_directory = f'{data_directory}/02_Docking'
isExist = os.path.exists(docking_data_directory)
if not isExist:
	os.makedirs(docking_data_directory)

data_analysis_data_directory = f'{data_directory}/03_Data_Analysis'
isExist = os.path.exists(data_analysis_data_directory)
if not isExist:
	os.makedirs(data_analysis_data_directory)

postp_data_directory = f'{data_directory}/04_post-proccessing'
isExist = os.path.exists(postp_data_directory)
if not isExist:
	os.makedirs(postp_data_directory)	

sdf_data_directory = f'{data_directory}/05_sdf_files'
isExist = os.path.exists(sdf_data_directory)
if not isExist:
	os.makedirs(sdf_data_directory)	

protein_prep_dir = f'{docking_directory}/protein_prep'
isExist = os.path.exists(protein_prep_dir)
if not isExist:
	os.makedirs(protein_prep_dir)
	


### START!


# 01 MOLECULE SELECTION WITH PHYSCHEM PROPRIETIES + DOUBLE BUTINA CLUSTERING

shutil.copy(f'{home_directory}/{full_db}', f'{mol_select_directory}/{full_db}') # move the library to the directory
os.chdir(f'{mol_select_directory}')


full_df = read_smiles_file(full_db)
full_df = full_df.dropna(subset=['smiles'])  # Drop rows where 'smiles' is NaN
full_df['smiles'] = full_df['smiles'].str.strip()
full_df['name'] = full_df['name'].str.strip()
full_df = full_df.drop_duplicates(subset=['smiles'])  # Remove duplicate SMILES entries
full_df = full_df.drop_duplicates(subset=['name'])  # Remove duplicate name entries
full_df = full_df.reset_index(drop=True)
sanitized_df = standardize_molecules(full_df) # sanitize the smiles and remove salts
sanitized_df.to_csv('standardized_molecules.smi', header = False, index = False, sep = ' ')
sanitized_df = sanitized_df.drop(columns=['mol','original_smiles'])
sanitized_df = sanitized_df[['smiles', 'name']]  # Reorder columns
length_full_df = len(sanitized_df)
full_database = Path(full_db).stem + '_cleaned.smi'
sanitized_df.to_csv(full_database, header = False, index = False, sep = ' ')

# Select the starting molecules based on the number of molecules in the database
if length_full_df > 2000000:
	print(f'\nThe input database contains {length_full_df} molecules, which is too many for the Butina clustering. \nA random selection will be performed\n')
	output_selection_df = sanitized_df.sample(n=int(length_full_df*0.01), random_state=0)
	output_selection_file = 'randomly_selected_molecules.smi'
	output_selection_df.to_csv(output_selection_file, header = False, index = False, sep = ' ')

elif length_full_df < 10000 and percentage < 0.1:
	if verbose == False:
		print(f'\nThe input database contains {length_full_df} molecules, start of the Butina clustering.\n')
		subprocess.run(["python", 'physchem_butina_sampling.py', full_database, str(0.1), 'n'], check=True)
		output_selection_file = 'physchem_butina_selected_molecules.smi'
	else:
		print(f'\nThe input database contains {length_full_df} molecules, start of the Butina clustering.\n')
		subprocess.run(["python", 'physchem_butina_sampling.py', full_database, str(0.1), 'v'], check=True)
		output_selection_file = 'physchem_butina_selected_molecules.smi'

else:
	print(f'\nThe input database contains {length_full_df} molecules, start of the Butina clustering.\n')
	if verbose == False:
		subprocess.run(["python", 'physchem_butina_sampling.py', full_database, str(percentage), 'n'], check=True)
	else:
		subprocess.run(["python", 'physchem_butina_sampling.py', full_database, str(percentage), 'v'], check=True)
	output_selection_file = 'physchem_butina_selected_molecules.smi'


os.chdir(f'{protein_prep_dir}')

# 02 PREPARE THE PROTEIN
shutil.copy(f'{home_directory}/{target_protein_file_input}', f'{protein_prep_dir}/{target_protein_file_input}')
shutil.copy(f'{docking_directory}/protein_prep.py', f'{protein_prep_dir}/protein_prep.py')

if verbose:
	subprocess.run([f'python protein_prep.py {target_protein_file_input}'], shell = True)
else: 
	subprocess.run([f'python protein_prep.py {target_protein_file_input} --quiet'] , shell = True)

target_protein_file = f'{Path(target_protein_file_input).stem}_prepared.pdb'

os.chdir(f'{docking_directory}')	
# create a  new directory where to put the selected molecules
docking_sel_mols = f"selected_molecules"
# Check whether the specified path exists or not
isExist = os.path.exists(docking_sel_mols)
if not isExist:
   # Create a new directory in case it does not exist
   os.makedirs(docking_sel_mols)

dock_sel_mols_dir = f'{docking_directory}/{docking_sel_mols}'

shutil.move(f'{mol_select_directory}/{output_selection_file}', f'{dock_sel_mols_dir}/{output_selection_file}')
shutil.copy(f'{protein_prep_dir}/{target_protein_file}', f'{dock_sel_mols_dir}/{target_protein_file}')
shutil.copy(f'{home_directory}/{target_ligand_file}', f'{dock_sel_mols_dir}/{target_ligand_file}')
shutil.copy('./remove_null.py', f'{dock_sel_mols_dir}')
shutil.copy('./smina_progress.py', f'{dock_sel_mols_dir}')

output_selection_file_stem = Path(output_selection_file).stem

# 03 DOCKING THE SELECTED MOLECULES

os.chdir(f'{dock_sel_mols_dir}')
# Convert to 3D
print('\n============================================ Docking ============================================')


print('\n=== Starting Parallel 3D Conversion and Docking for ROUND 0 ===')

try:
	# Run the parallel workflow
	start_sdf_file = parallel_docking_workflow(
		output_to_redock=output_selection_file,
		target_protein_file=target_protein_file,
		target_ligand_file=target_ligand_file,
		cpu=cpu,
		num_chunks=docking_chunks,
		cleanup_intermediate=True
	)
	
	print('\nDocking completed!\n')
except Exception as e:
	print(f'\nFATAL ERROR: The docking process has been terminated.')
	print(f'Reason: {str(e)}')


	
print('\nDocking of the selected molecules completed!\n')
	

shutil.copy(f'{dock_sel_mols_dir}/{start_sdf_file}', f'{home_directory}/{start_sdf_file}')
shutil.copy(f'{dock_sel_mols_dir}/{start_sdf_file}', f'{sdf_directory}/{start_sdf_file}')
shutil.copy(f'{protein_prep_dir}/{target_protein_file}', f'{docking_directory}/{target_protein_file}')
shutil.copy(f'{home_directory}/{target_ligand_file}', f'{docking_directory}/{target_ligand_file}')



# 04 MACHINE LEARNING


# move the files in the ml directory
shutil.move(f'{home_directory}/{start_sdf_file}', f'{ml_directory}/{start_sdf_file}')
shutil.copy(f'{mol_select_directory}/{full_database}', f'{ml_directory}/{full_database}')

# let's move into the ML directory
os.chdir(f'{ml_directory}')

initial_sdf_file = [file for file in glob.glob("*.sdf")][0]
initial_length = get_length(initial_sdf_file)
vprint(f'\nThe initial number of docked molecules is: {initial_length}')
selected_tasks_to_run = []

########### ROUNDS ###############


for i in range(rounds):
	# define the number of rounds
	r = i+1
	
	print(f'\n~~~~~~~~~~~~~~~~~~~~~~ROUND {r} START~~~~~~~~~~~~~~~~~~~~~~')

	# 1. Convert the sdf file to csv
	if r == 1:
	
		sdf_file = [file for file in glob.glob("*.sdf")][0]
		vprint(f'Converting {sdf_file} to csv')
	
		csv_file = sdf_to_csv(sdf_file)

		shutil.copy(f'{ml_directory}/{csv_file}', f'{data_viz_directory}/all_mean_redocked_mols_rank_by_rank_round_0.csv') # to do some dataviz later on		
		
	else:
		csv_file = [file for file in glob.glob("*.csv")][0]
		
	vprint(f'Using file {csv_file}')
	interim_file_df = pd.read_csv(csv_file, sep = ',', names = ['SMILES', 'ID', 'minimizedAffinity'], header = 0, dtype={'SMILES': str, 'ID': str, 'minimizedAffinity': float})

	vprint(f'Processed molecules for round {r}: {len(interim_file_df)}')

	# 2. Prep the database for ML training

	train_test_file, remaining_mols_file = prep_database_for_ML_training(csv_file, full_database)
	prep_docked_output_file = f'{Path(csv_file).stem}_MLtraining_ready.smi'
	prep_rem_output_file = f'{Path(full_database).stem}_MLtraining_remaining_mols.smi'
	os.rename(train_test_file, f'{Path(train_test_file).stem}_r{r}.smi')
	os.rename(remaining_mols_file, f'{Path(remaining_mols_file).stem}_r{r}.smi')
	train_test_file = f'{Path(train_test_file).stem}_r{r}.smi'
	remaining_mols_file = f'{Path(remaining_mols_file).stem}_r{r}.smi'
	
	
	shutil.move(f'{ml_directory}/{csv_file}', f'{mol_select_directory}/{csv_file}') # move the selected molecules back

######################################################  ML  #########################################################################
	tasks_to_run = []	
	
	if model != 'auto':
		tasks_to_run = model
		output_files = {}

	elif model == 'auto' and r == 1:
		tasks_to_run = ['rf', 'ri', 'xgb', 'svr', 'knr']
		output_files = {}
	
	elif model == 'auto' and r > 1:
		tasks_to_run = selected_tasks_to_run
		output_files = {}

	# 3.1 RandomForestRegressor
	if 'rf' in tasks_to_run:
		print('\n======================================== Random Forest =======================================')
		rf_directory = f'{ml_directory}/RF'
		rf_pred_output_file, rf_metrics_file, r2_train_rf, r2_val_rf = run_rf_training(ml_directory, data_analysis_directory, train_test_file, remaining_mols_file, full_database, r, cpu)
		shutil.move(f'{rf_directory}/{rf_pred_output_file}', f'{data_analysis_directory}/{rf_pred_output_file}') # move the prediction file to the data analysis directory
		shutil.move(f'{rf_directory}/{rf_metrics_file}', f'{data_analysis_directory}/{rf_metrics_file}') # move the metrics file to the data analysis directory
		output_files['rf_pred_output_file'] = rf_pred_output_file
		output_files['rf_metrics_file'] = rf_metrics_file

	# 3.2 Ridge regression
	if 'ri' in tasks_to_run:
		print('\n============================================ Ridge ===========================================')
		ridge_directory = f'{ml_directory}/ridge'
		ridge_pred_output_file, ridge_metrics_file, r2_train_ridge, r2_val_ridge = run_ridge_training(ml_directory, data_analysis_directory, train_test_file, remaining_mols_file, full_database, r, cpu)
		shutil.move(f'{ridge_directory}/{ridge_pred_output_file}', f'{data_analysis_directory}/{ridge_pred_output_file}') # move the prediction file to the data analysis directory
		shutil.move(f'{ridge_directory}/{ridge_metrics_file}', f'{data_analysis_directory}/{ridge_metrics_file}') # move the metrics file to the data analysis directory
		output_files['ridge_pred_output_file'] = ridge_pred_output_file
		output_files['ridge_metrics_file'] = ridge_metrics_file

	if 'xgb' in tasks_to_run:
		print('\n=========================================== XGBoost ==========================================')
		xgb_directory = f'{ml_directory}/xgboost'
		xgb_pred_output_file, xgb_metrics_file, r2_train_xgb, r2_val_xgb = run_xgb_training(ml_directory, data_analysis_directory, train_test_file, remaining_mols_file, full_database, r, cpu)
		shutil.move(f'{xgb_directory}/{xgb_pred_output_file}', f'{data_analysis_directory}/{xgb_pred_output_file}') # move the prediction file to the data analysis directory
		shutil.move(f'{xgb_directory}/{xgb_metrics_file}', f'{data_analysis_directory}/{xgb_metrics_file}') # move the metrics file to the data analysis directory
		output_files['xgb_pred_output_file'] = xgb_pred_output_file
		output_files['xgb_metrics_file'] = xgb_metrics_file
	
	if 'svr' in tasks_to_run:
		print('\n============================================= SVR ============================================')
		svr_directory = f'{ml_directory}/SVR'
		svr_pred_output_file, svr_metrics_file, r2_train_svr, r2_val_svr = run_svr_training(ml_directory, data_analysis_directory, train_test_file, remaining_mols_file, full_database, r, cpu)
		shutil.move(f'{svr_directory}/{svr_pred_output_file}', f'{data_analysis_directory}/{svr_pred_output_file}') # move the prediction file to the data analysis directory
		shutil.move(f'{svr_directory}/{svr_metrics_file}', f'{data_analysis_directory}/{svr_metrics_file}') # move the metrics file to the data analysis directory
		output_files['svr_pred_output_file'] = svr_pred_output_file
		output_files['svr_metrics_file'] = svr_metrics_file

	if 'knr' in tasks_to_run:
		print('\n============================================= KNR ============================================')
		knr_directory = f'{ml_directory}/KNR'
		knr_pred_output_file, knr_metrics_file, r2_train_knr, r2_val_knr = run_knr_training(ml_directory, data_analysis_directory, train_test_file, remaining_mols_file, full_database, r, cpu)
		shutil.move(f'{knr_directory}/{knr_pred_output_file}', f'{data_analysis_directory}/{knr_pred_output_file}') # move the prediction file to the data analysis directory
		shutil.move(f'{knr_directory}/{knr_metrics_file}', f'{data_analysis_directory}/{knr_metrics_file}') # move the metrics file to the data analysis directory
		output_files['knr_pred_output_file'] = knr_pred_output_file
		output_files['knr_metrics_file'] = knr_metrics_file	
				
	# if model is auto, select the best performing models based on the metrics from the first round on
	if model == 'auto' and r == 1:

		model_r2_val_dict = {'rf': r2_val_rf, 'ri': r2_val_ridge, 'xgb': r2_val_xgb, 'svr': r2_val_svr, 'knr': r2_val_knr}
		model_r2_train_dict = {'rf': r2_train_rf, 'ri': r2_train_ridge, 'xgb': r2_train_xgb, 'svr': r2_train_svr, 'knr': r2_train_knr}
		# to prevent errors in case one model failed. Namely R2 = 1
		bad_r2 = set()
		for k,v in model_r2_val_dict.items():
			if v == 1:
				bad_r2.add(k)
				print(f'Warning: Model {k} returned R2 = 1 in the validation set (failure), it will be excluded from the selection process.')

		# exclude models that overfitted (i.e. R2 train = 1)
		for k,v in model_r2_train_dict.items():
			if round(v, 2) >= 0.90: # round it beacuse even when it's close to 1 it means overfitting
				bad_r2.add(k)
				print(f'Warning: Model {k} returned R2 = 1 over the training set (overfitting), it will be excluded from the selection process.')

		bad_d = {k : model_r2_val_dict[k] for k in bad_r2}
		for k in bad_r2:
			model_r2_val_dict.pop(k)

		ordered_model_r2 = dict(sorted(model_r2_val_dict.items(), key=lambda item: item[1], reverse=True))

		if number_of_models > len(ordered_model_r2):
			number_of_models = len(ordered_model_r2)
			print(f'\nNote: The number of models to select has been adjusted to {number_of_models} due to model failures in round 1.\n')

		models_auto_selected = list(ordered_model_r2.keys())[:number_of_models]
		vprint(ordered_model_r2)
		vprint(models_auto_selected)
		print(f'\nModels selected for round 2 based on R2 values from round 1: {models_auto_selected}\n')
		selected_tasks_to_run = models_auto_selected


###############################################################################################################################

	# 4. Consensus scoring

	print('\n======================================== Consensus scoring =======================================')

	strategy = 'rank_by_rank' 

	os.chdir(f'{data_analysis_directory}')
	# create a new directory where to put the split molecules in
	path_r = f"round_{r}"

	isExist = os.path.exists(path_r)
	if not isExist:
		os.makedirs(path_r)

	# create a  new directory where to put metrics of the models in
	metrics_path_r = f"metrics"
	isExist = os.path.exists(metrics_path_r)
	if not isExist:
		os.makedirs(metrics_path_r)
	
	round_path = f'{data_analysis_directory}/{path_r}'
	metrics_path = f'{data_analysis_directory}/{metrics_path_r}'	
	
	os.chdir(f'{round_path}')

	# Move prediction files to the round-specific directory for consensus scoring
	files = []
	if model == 'auto' and r == 1:
		for mod in selected_tasks_to_run:
			if mod == 'rf':
				pred_f = output_files['rf_pred_output_file']
				metrics_f = output_files['rf_metrics_file']
				shutil.move(f'{data_analysis_directory}/{pred_f}', f'{round_path}/{pred_f}')
				shutil.move(f'{data_analysis_directory}/{metrics_f}', f'{metrics_path}/{metrics_f}')
				files.append(f'{round_path}/{pred_f}')
			if mod == 'ri':
				pred_f = output_files['ridge_pred_output_file']
				metrics_f = output_files['ridge_metrics_file']
				shutil.move(f'{data_analysis_directory}/{pred_f}', f'{round_path}/{pred_f}')
				shutil.move(f'{data_analysis_directory}/{metrics_f}', f'{metrics_path}/{metrics_f}')
				files.append(f'{round_path}/{pred_f}')
			if mod == 'xgb':
				pred_f = output_files['xgb_pred_output_file']
				metrics_f = output_files['xgb_metrics_file']
				shutil.move(f'{data_analysis_directory}/{pred_f}', f'{round_path}/{pred_f}')
				shutil.move(f'{data_analysis_directory}/{metrics_f}', f'{metrics_path}/{metrics_f}')
				files.append(f'{round_path}/{pred_f}')
			if mod == 'svr':
				pred_f = output_files['svr_pred_output_file']
				metrics_f = output_files['svr_metrics_file']
				shutil.move(f'{data_analysis_directory}/{pred_f}', f'{round_path}/{pred_f}')
				shutil.move(f'{data_analysis_directory}/{metrics_f}', f'{metrics_path}/{metrics_f}')
				files.append(f'{round_path}/{pred_f}')
			if mod == 'knr':
				pred_f = output_files['knr_pred_output_file']
				metrics_f = output_files['knr_metrics_file']
				shutil.move(f'{data_analysis_directory}/{pred_f}', f'{round_path}/{pred_f}')
				shutil.move(f'{data_analysis_directory}/{metrics_f}', f'{metrics_path}/{metrics_f}')
				files.append(f'{round_path}/{pred_f}')
	else:
		if 'rf_pred_output_file' in output_files:
			pred_f = output_files['rf_pred_output_file']
			metrics_f = output_files['rf_metrics_file']
			shutil.move(f'{data_analysis_directory}/{pred_f}', f'{round_path}/{pred_f}')
			shutil.move(f'{data_analysis_directory}/{metrics_f}', f'{metrics_path}/{metrics_f}')
			files.append(f'{round_path}/{pred_f}')
		if 'ridge_pred_output_file' in output_files:
			pred_f = output_files['ridge_pred_output_file']
			metrics_f = output_files['ridge_metrics_file']
			shutil.move(f'{data_analysis_directory}/{pred_f}', f'{round_path}/{pred_f}')
			shutil.move(f'{data_analysis_directory}/{metrics_f}', f'{metrics_path}/{metrics_f}')
			files.append(f'{round_path}/{pred_f}')
		if 'xgb_pred_output_file' in output_files:
			pred_f = output_files['xgb_pred_output_file']
			metrics_f = output_files['xgb_metrics_file']
			shutil.move(f'{data_analysis_directory}/{pred_f}', f'{round_path}/{pred_f}')
			shutil.move(f'{data_analysis_directory}/{metrics_f}', f'{metrics_path}/{metrics_f}')
			files.append(f'{round_path}/{pred_f}')
		if 'svr_pred_output_file' in output_files:
			pred_f = output_files['svr_pred_output_file']
			metrics_f = output_files['svr_metrics_file']
			shutil.move(f'{data_analysis_directory}/{pred_f}', f'{round_path}/{pred_f}')
			shutil.move(f'{data_analysis_directory}/{metrics_f}', f'{metrics_path}/{metrics_f}')
			files.append(f'{round_path}/{pred_f}')
		if 'knr_pred_output_file' in output_files:
			pred_f = output_files['knr_pred_output_file']
			metrics_f = output_files['knr_metrics_file']
			shutil.move(f'{data_analysis_directory}/{pred_f}', f'{round_path}/{pred_f}')
			shutil.move(f'{data_analysis_directory}/{metrics_f}', f'{metrics_path}/{metrics_f}')
			files.append(f'{round_path}/{pred_f}')

	if len(files) == 1:
		# If there is only one model, just sort the prediction file.
		model_name = Path(files[0]).stem.split('_')[0]
		output_file_consensus_scoring = f'{model_name}_scoring_results_r{r}.csv'
		shutil.copy(files[0], output_file_consensus_scoring)
		consensus_scoring_df = pd.read_csv(output_file_consensus_scoring, header=0, names=['smiles', 'name', 'PredictedScore'])
		consensus_scoring_df = consensus_scoring_df.sort_values(by='PredictedScore', ascending=True, ignore_index=True)
		
		# Select top molecules for the next round
		mols_to_redock_df = consensus_scoring_df.head(initial_length)
		mols_to_redock_df = mols_to_redock_df[['smiles', 'name']]

	elif len(files) > 1:
		# If there are multiple models, calculate consensus score.
		consensus_scoring_df = calculate_consensus_score(files, strategy)
		
		# Create a descriptive output filename
		model_names_str = '_'.join(sorted([Path(f).stem.split('_')[0] for f in files]))
		output_file_consensus_scoring = f'consensus_scoring_results_{model_names_str}_{strategy}_r{r}.csv'
		consensus_scoring_df.to_csv(output_file_consensus_scoring, index=False, sep=',')
		
		# Select top molecules for the next round
		mols_to_redock_df = consensus_scoring_df.head(initial_length)
		mols_to_redock_df = mols_to_redock_df[['smiles', 'name']]

	# Save the selected molecules for redocking
	mols_to_redock_df.to_csv(f'top_molecules_identified_in_round_{r}.csv', header=False, index=False, sep=',')
	
	# Take only the molecules that are NOT already in the docked file (the one that we will be keeping adding molecules to)
	
	shutil.copy(f'{mol_select_directory}/{csv_file}', f'{round_path}/{csv_file}') # big file that you add things to and from which you have to subtract
	print(f'Updating the file {csv_file} with the new molecules identified in round {r}')
	df1 = mols_to_redock_df.copy() # the ones that I need to subtract from the big file
	df2 = pd.read_csv(csv_file, header = 0, names = ['smiles', 'name', 'minimizedAffinity'], dtype={'smiles': str, 'name': str, 'minimizedAffinity': float}) # the big file

	# Merge and keep only rows in df1 that are NOT in df2 (based on 'id')
	vprint(f'Merging {csv_file} with {output_file_consensus_scoring} and removing overlapping molecules')
	merged = df1.merge(df2, on=['name'], how='left', indicator=True)
	result = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge', 'smiles_y'])
	result = result.rename(columns = {'smiles_x': 'smiles'})
	result = result.reset_index(drop = True)
	
	# Export the molecules to redock 
	output_to_redock = f'mols_to_redock_{strategy}_r{r}.smi' # these are the molecules to dock!
	output_to_redock_stem = Path(output_to_redock).stem
	result.to_csv(output_to_redock, header = False, index = False, sep = ' ')

	# Move the file into the docking directory
	shutil.move(f'{round_path}/{output_to_redock}', f'{docking_directory}/{output_to_redock}')

	print(f'\nConsensus scoring done \nNumber of molecules to redock: {len(result)}')
	vprint(f'Molecules to dock: {output_to_redock}\n')

	# move to the docking directory
	os.chdir(f'{docking_directory}')
	
	# create a  new directory where to put the split molecules
	docking_path_r = f"round_{r}"
	# Check whether the specified path exists or not
	isExist = os.path.exists(docking_path_r)
	if not isExist:
		# Create a new directory because it does not exist
		os.makedirs(docking_path_r)

	round_docking_path = f'{docking_directory}/{docking_path_r}'

	shutil.move(f'{docking_directory}/{output_to_redock}', f'{round_docking_path}/{output_to_redock}')
	shutil.copy('./remove_null.py', f'{round_docking_path}')
	shutil.copy('./smina_progress.py', f'{round_docking_path}')
	shutil.copy(f'./{target_protein_file}', f'{round_docking_path}')
	shutil.copy(f'./{target_ligand_file}', f'{round_docking_path}')

	os.chdir(f'{round_docking_path}')


	# 5. Docking

	# convert
	print(f'\n============================================ Docking ROUND {r} ============================================')
	  
	
	try:
		# Run the parallel workflow
		docking_output_file_round = parallel_docking_workflow(
			output_to_redock=output_to_redock,
			target_protein_file=target_protein_file,
			target_ligand_file=target_ligand_file,
			r=r,
			strategy = strategy,
			cpu=cpu,
			num_chunks=docking_chunks, 
			cleanup_intermediate=True
		)
	
		print('\nDocking completed!\n')
	except Exception as e:
		print(f'\nFATAL ERROR: The docking process has been terminated.')
		print(f'Reason: {str(e)}')
	
	
	# move the sdf file to the sdf directory to join them all at the end
	shutil.copy(f'{round_docking_path}/{docking_output_file_round}', f'{sdf_directory}/{docking_output_file_round}')

	
	# 6. Add the molecules that we've found to the initial file
	
	shutil.copy(f'{mol_select_directory}/{csv_file}', f'{round_docking_path}/{csv_file}') # file that we need to update
	round_before = pd.read_csv(csv_file, header = 0, names = ['smiles', 'name', 'minimizedAffinity'], dtype={'smiles': str, 'name': str, 'minimizedAffinity': float})

	docking_csv_file = sdf_to_csv(docking_output_file_round)
	docking_csv_file_read = pd.read_csv(docking_csv_file,  header = 0, names = ['smiles', 'name', 'minimizedAffinity'], dtype={'smiles': str, 'name': str, 'minimizedAffinity': float})
	
	not_overlapping = pd.concat([docking_csv_file_read, round_before], axis = 0)
	not_overlapping = not_overlapping.drop_duplicates(keep='first')
	next_round_file = f'redocked_molecules_added_{strategy}_round_{r}.csv'
	not_overlapping.to_csv(next_round_file, header = False, index = False)

	
	shutil.copy(f'{round_docking_path}/{next_round_file}', f'{data_viz_directory}/{next_round_file}') # to do some dataviz later
	
	if r == rounds:
		print('\nAll rounds completed!')		
	else:
		shutil.copy(f'{round_docking_path}/{next_round_file}', f'{ml_directory}') # to continue to the next round	
	os.chdir(f'{ml_directory}')	
	print(f'\n~~~~~~~~~~~~~~~~~~~~~~ROUND {r} COMPLETED~~~~~~~~~~~~~~~~~~~~~~')


# Clean up some of the directories ny putting all the files generated by the workflow in the data directory
clean_directory(docking_directory, docking_data_directory)
clean_directory(mol_select_directory, mol_select_data_directory)
clean_directory_no_dir(ml_directory, ml_data_directory)

tasks_to_run = model

output_files = {}

if tasks_to_run == 'auto':
	clean_directory(rf_directory, rf_data_directory)
	clean_directory(ridge_directory, ridge_data_directory)
	clean_directory(xgb_directory, xgboost_data_directory)
	clean_directory(svr_directory, svr_data_directory)
	clean_directory(knr_directory, knr_data_directory)
elif 'rf' in tasks_to_run:
	clean_directory(rf_directory, rf_data_directory)
elif 'ri' in tasks_to_run:
	clean_directory(ridge_directory, ridge_data_directory)
elif 'xgb' in tasks_to_run:
	clean_directory(xgb_directory, xgboost_data_directory)
elif 'svr' in tasks_to_run:
	clean_directory(svr_directory, svr_data_directory)
elif 'knr' in tasks_to_run:
	clean_directory(knr_directory, knr_data_directory)

# Create a single csv file that regroups the metrics of the ML methods in each round	
os.chdir(f'{metrics_path}')	
flist_m = []
all_files_m = [f for f in glob.glob("*.csv")]
for filename in all_files_m:
	df = pd.read_csv(filename, index_col=False)  
	flist_m.append(df)

df_out_m = pd.concat(flist_m, axis=0, ignore_index=False).sort_values(by= 'Unnamed: 0')
df_out_m.rename(columns = {'Unnamed: 0': 'Rounds'}, inplace = True)
df_out_m.to_csv("ML_metrics_per_round.csv", index = False, sep = ',')


# 03 DATA ANALYSIS AND VISUALIZATION

os.chdir(f'{data_viz_directory}')	
files = [file for file in glob.glob("*.csv")]
dataviz_length = int(initial_length * 0.96) # so that the visualization accounts for the same number of molecules

# here are some useful functions for dataviz
colors = ['#FF0000', '#FF8C00', '#800080', '#006400', '#0000CD', '#FFD700', '#00FFFF', '#008080', '#6495ED', '#32CD32','#191970', '#FF00FF', '#8B4513', '#708090', "#541010"]
selected_cols = colors[:len(files)]


file_col_dic = dictionary = dict(zip(files, selected_cols))
file_col_dic_od = collections.OrderedDict(sorted(file_col_dic.items()))

def get_round_num(f):
	'''
	Extract the round number from the filename
	'''
	round_num = int((Path(f).stem.split('_')[-1]))
	return round_num
round_dic = {get_round_num(f): f for f in files}
round_dic = collections.OrderedDict(sorted(round_dic.items()))

def read_docking_score_distribution_sing(input_file, color):
	'''
	Plot the distribution of docking scores from a single file
	'''
	df = pd.read_csv(input_file, header = 0, names = ['smiles', 'name', 'minimizedAffinity'], dtype={'smiles': str, 'name': str, 'minimizedAffinity': float})
	df.sort_values(by = 'minimizedAffinity', ignore_index = True, inplace = True)
	df = df[:dataviz_length]
	data = df['minimizedAffinity']
	name = Path(input_file).stem.split('_')[-1]
	bins = 100
	myHist = plt.hist(data, bins, density=True, alpha=0.4, edgecolor='none', color = color)
	kde = sps.gaussian_kde(data)
	ax = plt.gca()
	plt.plot(np.sort(data), kde.pdf(np.sort(data)), label=f'{name}', c = color)
	# set axis labels
	ax.set_xlabel('Minimized Affinity', fontsize=12)
	ax.set_ylabel('Frequency', fontsize=12)
	plt.title('Docking scores distribution', fontdict=None, loc='center')
	plt.legend();


def read_docking_score_distribution_top_mols(input_file, color):
	'''
	Plot the distribution of docking scores from a single file, plotting it into bigger bins to see the details of the best scoring molecules
	'''
	df = pd.read_csv(input_file, header = 0, names = ['smiles', 'name', 'minimizedAffinity'], dtype={'smiles': str, 'name': str, 'minimizedAffinity': float})
	df.sort_values(by = 'minimizedAffinity', ignore_index = True, inplace = True)
	data = df['minimizedAffinity']
	min_d = data.min()
	bins = 30
	
	# create figure and axis
	fig, ax = plt.subplots(figsize=(10, 6))
	
	counts, edges, bars = ax.hist(data, bins, alpha=0.4, color=color, edgecolor='black', linewidth=1.2)
	#ax.set_xlim([(min_d-0.5), (min_d/1.3)])
	
	# set axis labels
	ax.set_xlabel('Minimized Affinity', fontsize=12)
	ax.set_ylabel('Frequency', fontsize=12)
	ax.tick_params(axis='both', which='major', labelsize=10)
	
	# bar labels
	ax.bar_label(bars)
	
	plt.title('Docking scores distribution', fontdict=None, loc='center')
	plt.tight_layout()  # helps with spacing
	plt.savefig(f'bigger_bins_distribution.png', dpi=300, bbox_inches='tight')
	plt.close()  # close the figure to free memory


	
def read_distributions(dic):
	'''
	Plot the distribution of docking scores from multiple files from a dictionary in the same plot
	'''
	for fil, color in dic.items():
		read_docking_score_distribution_sing(fil, color)
	plt.savefig('mean_distributions.png')


def get_mean_sd_from_file(input_file):
	'''
	Extract the mean, the sd, the min and the max from a file
	'''
	round_num = get_round_num(input_file)
	df = pd.read_csv(input_file, sep = ',', header = 0, names = ['smiles', 'name', 'minimizedAffinity'], dtype={'smiles': str, 'name': str, 'minimizedAffinity': float})
	df = df.sort_values(by = 'minimizedAffinity', ignore_index = True)
	df = df[:dataviz_length]
	df_mean = df['minimizedAffinity'].mean()
	df_sd = df['minimizedAffinity'].std()
	df_min = df['minimizedAffinity'].min()
	df_max = df['minimizedAffinity'].max()

	return round_num, df_mean, df_sd, df_min, df_max

def plot_mean_variation(df):
	'''
	Plot the variation of the mean of every round with error bars
	'''
	df = df[:dataviz_length]
	plt.figure(figsize=(10, 10))
	x = np.array(df.index.values.tolist())
	y = np.array(df['Mean'])
	error = np.array(df['SD'])
	fig, ax = plt.subplots()
	ax.set_xlabel('Rounds')
	ax.set_ylabel('Mean value')
	ax.errorbar(x, y, yerr=error, fmt='-o', capsize=3)
	ax.set_title('Mean variation')

	plt.savefig('mean_variation.png');
	

def plot_boxplot(dic):
	'''
	Create a boxplot using seaborn with proper data formatting
	'''
	# Combine all data into a single DataFrame
	combined_data = []
	color_mapping = {}
	
	for i, (n, f) in enumerate(dic.items()):
		df = pd.read_csv(f, header=0, names=['smiles', 'name', 'minimizedAffinity'], dtype={'smiles': str, 'name': str, 'minimizedAffinity': float})
		df = df.sort_values(by='minimizedAffinity', ignore_index=True)
		
		# Extract numeric round number
		round_num = int(str(n).replace('r', '')) if 'r' in str(n) else n

		# Add round column
		df['Round'] = round_num
		combined_data.append(df[['minimizedAffinity', 'Round']])
		
		# Map round to color (use the same colors as your original scheme)
		color_mapping[round_num] = colors[i] if i < len(colors) else colors[-1]
	
	# Concatenate all dataframes
	final_df = pd.concat(combined_data, ignore_index=True)
	
	# Sort by round number for proper ordering
	final_df = final_df.sort_values('Round')
	
	# Concatenate all dataframes
	unique_rounds = sorted(final_df['Round'].unique())
	palette = [color_mapping[round_num] for round_num in unique_rounds]
	
	# Create boxplot
	plt.figure(figsize=(10, 6))
	sns.boxplot(data=final_df, x='Round', y='minimizedAffinity', hue='Round', palette=palette, legend=False)
	plt.xlabel('Rounds')
	plt.ylabel('Minimized Affinity')
	plt.title('Docking Scores Distribution by Round')
	plt.tight_layout()
	plt.savefig('boxplot.png')
	plt.close()
	

# Save the statistics
stats_data_dic = {}
for f in files:
	round_num, mean, sd, minn, maxx = get_mean_sd_from_file(f)
	stats_data_dic[round_num] = [round(mean, 2), round(sd, 2), round(minn, 2), round(maxx, 2)]
od = collections.OrderedDict(sorted(stats_data_dic.items()))
stats_per_round = pd.DataFrame(od).transpose() 
stats_per_round.rename(columns = {0: 'Mean', 1: 'SD', 2: 'min', 3: 'max'}, inplace = True)
stats_per_round.to_csv('mean_per_round.csv', index = True, sep = ',')
shutil.move(f'./mean_per_round.csv', f'{data_analysis_data_directory}/mean_per_round.csv')

# save the distribution image
read_distributions(file_col_dic_od)

# save the plot of the variation of the mean of every round
plot_mean_variation(stats_per_round)
plt.clf()

# create the boxplot
plot_boxplot(round_dic)

# explore the last distribution in more detail, diving it into bigger bins
last_file = round_dic.popitem()[1]
read_docking_score_distribution_top_mols(last_file, selected_cols[-1])



# Create the sdf containing all the molecules that we have docked
os.chdir(f'{sdf_directory}')	
complete_docked_mols = f'all_docked_molecules_{rounds}_rounds.sdf'
subprocess.run([f'cat *.sdf >> {complete_docked_mols}'], shell=True, text = True, bufsize = 1)
shutil.copy(f'{sdf_directory}/{complete_docked_mols}', f'{home_directory}/{complete_docked_mols}')
print(f'Full docked molecules saved as {complete_docked_mols}')
clean_directory(sdf_directory, sdf_data_directory)
clean_directory(data_analysis_directory, data_analysis_data_directory)



# POST-PROCESSING
os.chdir(f'{post_processing_directory}')
shutil.copy(f'{sdf_data_directory}/{complete_docked_mols}', f'{post_processing_directory}/{complete_docked_mols}')
shutil.copy(f'{docking_data_directory}/{target_protein_file}', f'{post_processing_directory}/{target_protein_file}')

print('\n======================================== Post-processing =======================================')

# sort the molecules to obtain the best scoring ones
vprint('\nSorting the docked molecules...')

FLOAT_RE = re.compile(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?')

def parse_score(block):
	"""Extract float score from minimizedAffinity field in an SDF block."""
	# Try multiple possible formats for the minimizedAffinity field
	patterns = [
		rf">\s*<\s*'minimizedAffinity'\s*>\s*\n([^\n\r$]+)",  # Original pattern with quotes
		rf">\s*<\s*minimizedAffinity\s*>\s*\n([^\n\r$]+)",   # Without quotes
		rf">\s*<minimizedAffinity>\s*\n([^\n\r$]+)",         # Simple format
		rf">\s*<\s*minimizedAffinity\s*>\s*\r?\n([^\n\r$]+)", # Handle different line endings
	]
	
	for pattern in patterns:
		m = re.search(pattern, block, re.IGNORECASE)
		if m:
			val = m.group(1).strip()
			# Remove any trailing $$ that might be captured
			val = val.split('$$')[0].strip()
			n = FLOAT_RE.search(val)
			if n:
				return float(n.group())
	
	return None

def debug_sdf_structure(in_sdf, num_blocks_to_check=5):
	"""Debug function to examine SDF structure"""
	vprint(f"Debugging SDF file: {in_sdf}")
	
	if not os.path.exists(in_sdf):
		print(f"ERROR: File {in_sdf} does not exist!")
		return
	
	file_size = os.path.getsize(in_sdf)
	vprint(f"File size: {file_size} bytes")
	
	try:
		with open(in_sdf, "r", encoding="utf-8", errors="replace") as f:
			content = f.read()
	except Exception as e:
		print(f"Error reading file: {e}")
		return
	
	# Check for different possible separators
	separators = ["$$$$", "$$$$\n", "\n$$$$\n", "\n$$$$"]
	for sep in separators:
		count = content.count(sep)
		vprint(f"Found {count} occurrences of separator '{repr(sep)}'")
	
	# Split into blocks using the most common separator
	blocks = [b.strip() for b in content.split("$$$$") if b.strip()]
	vprint(f"Total blocks found: {len(blocks)}")
	
	if len(blocks) == 0:
		print("No blocks found! Check your SDF file format.")
		return
	
	# Examine first few blocks
	for i, block in enumerate(blocks[:num_blocks_to_check]):
		vprint(f"\n--- Block {i+1} ---")
		vprint(f"Block length: {len(block)} characters")
		
		# Look for property fields
		property_fields = re.findall(r'>\s*<([^>]+)>', block)
		vprint(f"Property fields found: {property_fields}")
		
		# Try to extract score
		score = parse_score(block)
		vprint(f"Extracted score: {score}")


def reorder_sdf(in_sdf, k=10000, higher_better=False):
	"""Reorder SDF file by minimizedAffinity score and extract top k molecules."""
	
	if not os.path.exists(in_sdf):
		vprint(f"ERROR: Input file {in_sdf} does not exist!")
		return None
	
	vprint(f"Processing file: {in_sdf}")
	vprint(f"Looking for top {k} molecules)")
	
	try:
		with open(in_sdf, "r", encoding="utf-8", errors="replace") as f:
			content = f.read()
	except Exception as e:
		vprint(f"Error reading file: {e}")
		return None
	
	out_sdf = f'{Path(in_sdf).stem}_top{k}.sdf'
	
	# Split into blocks (remove any trailing empty parts)
	blocks = [b.strip() for b in content.split("$$$$") if b.strip()]
	vprint(f"Total blocks found: {len(blocks)}")
	
	if len(blocks) == 0:
		vprint("No blocks found in SDF file!")
		return None
	
	scored = []
	no_score_count = 0
	
	for i, block in enumerate(blocks):
		sc = parse_score(block)
		if sc is None:
			no_score_count += 1
			continue
		
		eff = -sc if higher_better else sc
		scored.append((eff, i, block, sc))
	
	vprint(f"Blocks with valid scores: {len(scored)}")
	vprint(f"Blocks without scores: {no_score_count}")
	
	if len(scored) == 0:
		print("ERROR: No blocks with valid minimizedAffinity scores found!")
		return None
	
	# Sort by effective score, then by index (for stability)
	scored.sort(key=lambda x: (x[0], x[1]))
	top = scored[:k]
	
	vprint(f"Extracting top {len(top)} molecules")
	
	if len(top) > 0:
		vprint(f"Best score: {top[0][3]}")
		vprint(f"Worst score in selection: {top[-1][3]}")
	
	# Write out blocks, adding back "$$"
	try:
		with open(out_sdf, "w", encoding="utf-8") as out:
			for _, _, block, score in top:
				out.write(block.strip() + "\n\n$$$$\n")
		
		vprint(f"Successfully wrote {len(top)} molecules to {out_sdf}")
		return out_sdf
		
	except Exception as e:
		print(f"Error writing output file: {e}")
		return None

if get_length(complete_docked_mols) < 50000:
	complete_docked_mols_ordered = reorder_sdf(complete_docked_mols, k=get_length(complete_docked_mols), higher_better=False)
else:
	complete_docked_mols_ordered = reorder_sdf(complete_docked_mols, k=50000, higher_better=False)

if not complete_docked_mols_ordered:
	print("ERROR: Reordering SDF failed!")

else:
	print(f'Post-processing: best {get_length(complete_docked_mols_ordered)} molecules')
	n_compounds = '10'
	subprocess.run(["python", 'post_processing_script.py', f'{complete_docked_mols_ordered}', '--protein', f'{target_protein_file}', '--top-n', n_compounds])

	print(f'\nPost-processing dashboard saved as {Path(complete_docked_mols_ordered).stem}_top_compounds.html')

# Extract the best molecules for further analysis
top_mols = f"{Path(complete_docked_mols_ordered).stem}_enhanced_analysis_top_{n_compounds}_compounds.sdf"


top_suppl = Chem.SDMolSupplier(top_mols)
top_mols_list = [mol for mol in top_suppl if mol is not None]

mols_for_shap = []

try:
	with Chem.SDWriter('first_mol.sdf') as w:
		w.write(top_mols_list[0])
		mols_for_shap.append('first_mol.sdf')
except Exception as e:
	print(f"Error writing first molecule: {e}")
shutil.copy(f'{post_processing_directory}/first_mol.sdf', f'{boostsfshap_directory}/first_mol.sdf')


try:
	with Chem.SDWriter('second_mol.sdf') as w:
		w.write(top_mols_list[1])
		mols_for_shap.append('second_mol.sdf')
except Exception as e:
	print(f"Skipping the second molecule: {e}")
shutil.copy(f'{post_processing_directory}/second_mol.sdf', f'{boostsfshap_directory}/second_mol.sdf')


print('\n~~~~~~~~~~~~~~~~~~~~~Post-processing completed! ~~~~~~~~~~~~~~~~~~~~~\n')

clean_directory_no_dir(post_processing_directory, postp_data_directory)



# Plug-ins execution
print('\n======================================== Plug-ins =======================================')

if 'boostsf-shap' in plug_in:
	print('~~~~~~~~~~~~~~~~~~~~~ BoostSF-SHAP ~~~~~~~~~~~~~~~~~~~~~')
	os.chdir(boostsfshap_directory)
	shutil.copy(f'{docking_data_directory}/protein_prep/{target_protein_file}', f'{boostsfshap_directory}/{target_protein_file}')
	shutil.copy(f'{home_directory}/{target_ligand_file}', f'{boostsfshap_directory}/{target_ligand_file}')
	pdbqt_prot = f'{Path(target_protein_file).stem}.pdbqt'
	pdbqt_lig = f'{Path(target_ligand_file).stem}.pdbqt'
	vprint('Coinverting target protein and ligand to PDBQT format for BoostSF-SHAP plug-in...')
	subprocess.run(["obabel", "-ipdb", f"{target_protein_file}", "-opdbqt", "-O", f"{pdbqt_prot}"], stderr = subprocess.DEVNULL, stdout = subprocess.DEVNULL)
	subprocess.run(["obabel", "-ipdb", f"{target_ligand_file}", "-opdbqt", "-O", f"{pdbqt_lig}"], stderr = subprocess.DEVNULL, stdout = subprocess.DEVNULL)

	for b in mols_for_shap:
		pdbqt_mol = f'{Path(b).stem}.pdbqt'
		vprint(f'Converting {b} to PDBQT format for BoostSF-SHAP plug-in...')
		subprocess.run(["obabel", "-isdf", f"{b}", "-opdbqt", "-O", f"{Path(b).stem}.pdbqt"])
		vprint(f'\nRunning BoostSF-SHAP plug-in for {pdbqt_mol}...')
		subprocess.run(['python', 'BoostSF-SHAP-main.py', '--modeltype', 'catboost', '--model', './models/catboost-BV/catBoost-V10_BV.cbm', '--receptor', f'{pdbqt_prot}', '--reference', f'{pdbqt_lig}', '--ligand', f"{pdbqt_mol}"])
		print(f'BoostSF-SHAP plug-in completed for {pdbqt_mol}!\n')
		print(f'Results saved in {boostsfshap_directory}/results-{Path(b).stem}"\n')


else:
	print('No plug-in selected')



# End timer
end_time = time.time()

# Calculate elapsed time
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")