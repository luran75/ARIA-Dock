import pickle
import pandas as pd
import numpy as np
import sys
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
from pathlib import Path

RDLogger.DisableLog('rdApp.*')  # Suppress RDKit warnings

def smiles_to_fingerprints(smiles_list, radius=2, n_bits=2048):
    fingerprints = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            fingerprints.append(np.array(fp))
        else:
            fingerprints.append(np.zeros(n_bits))  # Handle invalid SMILES
    return np.array(fingerprints)

# Function to load the model and its information
def load_model_and_info(model_filename):
    with open(model_filename, "rb") as f:
        loaded_model, model_info = pickle.load(f)  # Load the model and saved info
    return loaded_model, model_info

def predict_external_molecules_from_file(model, file_path):
    try:
        external_data = pd.read_csv(file_path, sep='\s+', header=None, names=["SMILES", "Name"]) 
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None  # Handle the error as needed
    except pd.errors.ParserError:
        print(f"Error: Could not parse the file at {file_path}. Check the file format.")
        return None

    external_smiles = external_data["SMILES"].tolist()
    fingerprints = smiles_to_fingerprints(external_smiles)
    scores = model.predict(fingerprints)
    external_data["PredictedScore"] = scores
    return external_data


    

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python mycode.py model_filename external_file round")
        sys.exit(1)
        
    model_filename = sys.argv[1]
    external_file = sys.argv[2] 
    r = sys.argv[3] 
    

    try:
        loaded_model, model_info = load_model_and_info(model_filename)
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_filename}")
        exit()  # Exit the script if the model isn't found
    except Exception as e: # Catch any other potential errors during loading
        print(f"Error loading model: {e}")
        exit()

    chunk_num = int(Path(external_file).stem.split('_')[-1])


    external_results = predict_external_molecules_from_file(loaded_model, external_file)

    if external_results is not None:  # Check if file loading and prediction were successful
        output_filename = f"RF_regression-grid-predicted-scores_c{chunk_num}_r{r}.csv"
        external_results.to_csv(output_filename, index=False)
    else:
        print("Prediction process failed. Check the error messages above.")

