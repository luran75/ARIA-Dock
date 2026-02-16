import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import re
import random
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import MACCSkeys  # Import MACCSkeys
from rdkit.Chem import AllChem, DataStructs
from rdkit.ML.Cluster import Butina
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import random
random.seed(42)
np.random.seed(42)

'''
This script performs Step 1 = physicochemical diversity sampling and Step 2 = a two-stage Butina structural clustering on chunks.
Output files: final molecules selected after Physchem and Butina clustering in Smiles format, 1 representative molecule per cluster
 
We run a two-stage Butina clustering on chunks to speed-up computations not based on real centroids.

Stage 1: Loose clustering
FIRST_CUTOFF = 0.60:
Distance ≤ 0.60 to be in same cluster

similarity threshold ≥ 0.40 
Stage 1 clusters molecules with ≥40% similarity with the selected fingerprints (relatively permissive)

Stage 2: Tight clustering
SECOND_CUTOFF = 0.3:
Distance ≤ 0.3 to be in same cluster

similarity threshold ≥ 0.7
Stage 2 clusters molecules with ≥70% similarity (relatively tight)
'''

def vprint(*args, **kwargs):
	"""Print only if verbose mode is enabled"""
	if verbose:
		print(*args, **kwargs)

#### STEP 1: PHYSCHEM CLUSTERING

def calculate_descriptors(smiles):
    """Vectorized function to compute physicochemical descriptors for a molecule."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return pd.Series([np.nan, np.nan, np.nan, np.nan], index=['MW', 'logP', 'RotBonds', 'TPSA'])
        return pd.Series([
            Descriptors.ExactMolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.TPSA(mol)
        ], index=['MW', 'logP', 'RotBonds', 'TPSA'])
    except:
        return pd.Series([np.nan, np.nan, np.nan, np.nan], index=['MW', 'logP', 'RotBonds', 'TPSA'])

def create_bin_label(row, bins_dict):
    """Generate bin labels for each molecule based on physicochemical bins."""
    try:
        return "_".join(f"{desc}_{np.digitize(row[desc], bins_dict[desc][1:-1])}" for desc in bins_dict)
    except:
        return None

def physchem_binned_sampling(input_file, n_samples, n_bins=5, min_class_size=2):
    """
    Performs Physicochemical Binned Sampling.
    
    The script bins molecules into categories (based on MW, logP, Rotatable Bonds, and TPSA).
    It samples molecules proportionally from each bin but does not ensure strict stratification like classic stratified sampling.
    Here, the bins are computed dynamically instead of being predefined classes.
    
    Parameters:
    - input_file: Path to input file (tab or space-separated, with SMILES & Names)
    - n_samples: Number of molecules to sample
    - n_bins: Number of bins for each descriptor (for example with 5 bins and MW values 0-20th percentile, 20-40th, 40-60th, 60-80th, 80-100th)
    - min_class_size: Minimum number of molecules per bin

    Returns:
    - DataFrame with sampled molecules
    """
    vprint(f" Starting sampling process from file: {input_file}")
    vprint(f" Parameters -> n_samples: {n_samples}, n_bins: {n_bins}, min_class_size: {min_class_size}\n")

    # Read input file (handles both tab and space-separated files)
    df = pd.read_csv(input_file, sep='\s+', header=None, names=['SMILES', 'Name'])
    
    print(f"Loaded dataset with {len(df)} molecules.\n")
    # Compute molecular descriptors
    print("Computing molecular descriptors... This may take some time for large datasets.")
    df[['MW', 'logP', 'RotBonds', 'TPSA']] = df['SMILES'].apply(calculate_descriptors)

    # Drop molecules with missing descriptors
    df.dropna(inplace=True)
    INITIAL_LENGTH = int(len(df))
    vprint(f"Descriptors computed. {INITIAL_LENGTH} molecules remain after filtering invalid SMILES.\n")

    # Create bins using percentiles
    vprint("Creating bins for each physicochemical property...")
    bins_dict = {desc: np.percentile(df[desc], np.linspace(0, 100, n_bins + 1)) for desc in ['MW', 'logP', 'RotBonds', 'TPSA']}

    # Assign bin labels
    df['BinLabel'] = df.apply(lambda x: create_bin_label(x, bins_dict), axis=1)

    # Filter out bins with fewer than `min_class_size` molecules
    valid_bins = df['BinLabel'].value_counts()[lambda x: x >= min_class_size].index
    df = df[df['BinLabel'].isin(valid_bins)]
    vprint(f"{len(valid_bins)} valid bins remain after filtering.\n")

    # Compute sampling weights
    class_weights = df['BinLabel'].value_counts(normalize=True)
    samples_per_class = (class_weights * n_samples).round().astype(int)

    # Perform grouped sampling
    vprint("Performing binned sampling...\n")
    
    # Previous pandas versions generate some warning in terminal
    sampled_df = df.groupby('BinLabel', group_keys=False).apply(
        lambda x: x.sample(n=min(len(x), samples_per_class.get(x.name, 0)))
    )

    # As the script calculates proportional weights for each bin, add a few molecules at random in case there are not enough molecules
    if len(sampled_df) < n_samples:
        remaining = n_samples - len(sampled_df)
        print(f"Not enough samples collected ({len(sampled_df)}/{n_samples}). Adding {remaining} random samples.")
        additional_samples = df[~df.index.isin(sampled_df.index)].sample(n=min(len(df), remaining))
        sampled_df = pd.concat([sampled_df, additional_samples])

    # Prepare final output
    output_df = sampled_df[['SMILES', 'Name', 'MW', 'logP', 'RotBonds', 'TPSA']]
            
    
    return output_df


    
#### STEP 2: BUTINA CLUSTERING    
# Read space separated or tab separated input data
def read_smiles(filename):
    smiles_list = []
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if line:  # Check if line is not empty
                try:
                    # Split by space or tab
                    parts = re.split(r'[ \t]', line, 1)
                    if len(parts) == 2:
                        smiles, name = parts
                        mol = Chem.MolFromSmiles(smiles)
                        if mol:
                            smiles_list.append((smiles, name))
                    else:
                        print(f"Skipping malformed line: {line}")
                except ValueError:
                    print(f"Skipping malformed line: {line}")
    return smiles_list


# STEP 3 Generate Morgan fingerprints
# Morgan radius 2 (captures 4-bond diameter substructures)
def generate_fingerprints(smiles_list):
    fingerprints = []
    mols = []
    for smiles, name in smiles_list: # Directly unpack tuple
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048, useChirality=False)
            fingerprints.append(fp)
            mols.append((smiles, name, mol))
    return fingerprints, mols

# STEP 4. Clustering Molecules Using Butina Algorithm

def cluster_fingerprints(fingerprints, cutoff, min_cluster_size=2):
    if not fingerprints:
        return [], []

    nfps = len(fingerprints)
    dists = []
    for i in range(1, nfps):
        sims = DataStructs.BulkTanimotoSimilarity(fingerprints[i], fingerprints[:i])
        # Calculate distances directly within the loop.
        for sim in sims:
            dists.append(1 - sim) # Append the calculated distance. We have 1 - similarity 

    clusters = Butina.ClusterData(dists, nfps, cutoff, isDistData=True)

    valid_clusters = [c for c in clusters if len(c) >= min_cluster_size]
    singletons = [c[0] for c in clusters if len(c) < min_cluster_size]
    vprint(f"Cutoff: {cutoff}, Number of Clusters: {len(valid_clusters)}, Number of Singletons should be 0 with the paramters used and the convention for the cluster-type names. Computed number: {len(singletons)}")  # Print No singletons, as wanted, because of parameter used
    return valid_clusters, singletons



def find_diverse_centroids(cluster, fingerprints, mols, max_centroids=5):
    if len(cluster) <= max_centroids:
        return [(mols[i][0], mols[i][1]) for i in cluster]

    cluster_fps = [fingerprints[i] for i in cluster]
    selected_indices = []

    # Calculate average similarity more efficiently
    sims_matrix = np.array([DataStructs.BulkTanimotoSimilarity(fp1, cluster_fps) for fp1 in cluster_fps])
    avg_sims = np.mean(sims_matrix, axis=1)
    first_idx = np.argmax(avg_sims)
    selected_indices.append(cluster[first_idx])

    while len(selected_indices) < max_centroids:
        best_idx = -1
        max_min_dist = -1

        for i in cluster:
            if i not in selected_indices:
                min_dist = 1 # Initialize with max distance.
                for j in selected_indices:
                    min_dist = min(min_dist, 1 - DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j]))
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_idx = i
        if best_idx != -1:
            selected_indices.append(best_idx)
    return [(mols[i][0], mols[i][1]) for i in selected_indices]


# STEP 6. Two-Stage Clustering Process

def two_stage_clustering(input_df, chunk_size=2500):
    FIRST_CUTOFF = 0.6 # default value 0.6
    SECOND_CUTOFF = 0.3
    MIN_CLUSTER_SIZE = 1 # if 1 then no singletons, to have singletons = MIN_CLUSTER_SIZE should be 2

    smiles_list = [(row['SMILES'], row['Name']) for _, row in input_df.iterrows()]
    vprint(f"Total molecules to cluster: {len(smiles_list)}")

    first_stage_centroids = []
    
    # First stage clustering
    for i in range(0, len(smiles_list), chunk_size):
        chunk = smiles_list[i:i + chunk_size]
        vprint(f"The size of the chunks are defined by the script. Processing chunk {i // chunk_size + 1} of {len(smiles_list)//chunk_size +1}")
        fingerprints, mols = generate_fingerprints(chunk)
        clusters, _ = cluster_fingerprints(fingerprints, FIRST_CUTOFF, MIN_CLUSTER_SIZE)

        for cluster in clusters:
            centroids = find_diverse_centroids(cluster, fingerprints, mols)
            first_stage_centroids.extend(centroids)

    vprint(f"First stage complete. Found {len(first_stage_centroids)} centroids-like molecules")

    # Second stage clustering
    vprint(f"Starting second stage with {len(first_stage_centroids)} molecules")

    final_representatives = []

    # Process second stage in chunks
    for i in range(0, len(first_stage_centroids), chunk_size):
        chunk = first_stage_centroids[i:i + chunk_size]
        vprint(f"Processing second stage chunk {i // chunk_size + 1} of {len(first_stage_centroids)//chunk_size +1}")
        
        fingerprints, mols = generate_fingerprints(chunk)
        clusters, _ = cluster_fingerprints(fingerprints, SECOND_CUTOFF, MIN_CLUSTER_SIZE)
        
        for cluster in clusters:
            # Select the first molecule as the representative for the final output
            representative = mols[cluster[0]]
            final_representatives.append(representative)

    print(f"Two stage clustering complete. Found {len(final_representatives)} representative molecules")

    # Create and return a DataFrame
    final_df = pd.DataFrame(final_representatives, columns=['SMILES', 'Name', 'Mol'])
    final_df.drop('Mol', axis=1, inplace=True)
    
    # Drop duplicates just in case
    final_df.drop_duplicates(subset=['SMILES'], inplace=True)
    
    return final_df




# STEP 7. Running the Script

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python this-script.py input_library percentage_to_sample verbose")
        sys.exit(1)

    verbose = True if sys.argv[3].lower() == 'v' else False
    vprint = lambda *args, **kwargs: print(*args, **kwargs) if verbose else None

    vprint('=== Physchem Clustering Step ===')
    input_file = sys.argv[1]
    
    df = pd.read_csv(input_file, sep='\s+', header=None, names=['SMILES', 'Name'])
    n_samples = int(float(sys.argv[2])*len(df)) # percentage of molecules to sample

    
    if len(df) < 1000:
        n_bins = 3
    elif 1000 <= len(df) < 100000:
        n_bins = 5
    else:
        n_bins = 7
        
    min_class_size = 2 # Min molecules per bin
    

    sampled_df = physchem_binned_sampling(input_file, n_samples, n_bins, min_class_size)

    vprint(f"Physico-chemical diversity sampling successful, keep {len(sampled_df)} molecules.")

    # Convert the sampled DataFrame to a simple SMILES/Name DataFrame for clustering
    df_for_butina = sampled_df[['SMILES', 'Name']].copy()
    
    vprint('=== Butina Clustering Step ===') 
    
    # Define chunk_size based on the number of molecules remaining after physchem binning
    if len(df_for_butina) < 10000:
        chunk_size = int(len(df_for_butina))
        #chunk_size = 500 #This was just for debug
    elif 10000 <= len(df_for_butina) < 100000:
        chunk_size = 3000
    else:
        chunk_size = 1500

    # Pass the DataFrame directly to the function
    final_df = two_stage_clustering(df_for_butina, chunk_size)

    # Save the final output to a file
    final_smi_file = 'physchem_butina_selected_molecules.smi'
    final_df.to_csv(final_smi_file, index=False, header=False, sep=' ')


    vprint(f"Diverse representatives saved to {final_smi_file}")
    vprint(f"~~~~ {len(final_df)} structurally and physicochemically diverse molecules selected, with one representative member per structural cluster ~~~~")


