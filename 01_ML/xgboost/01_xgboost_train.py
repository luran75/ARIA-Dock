import pickle
import argparse
import pandas as pd
import numpy as np
from rdkit import Chem
from sklearn.model_selection import KFold
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator
from rdkit import RDLogger
import optuna
from joblib import Parallel, delayed
from tqdm import tqdm

# to avoid rdkit warning message in the terminal
RDLogger.DisableLog('rdApp.*')

def vprint(*args, **kwargs):
	"""Print only if verbose mode is enabled"""
	if verbose:
		print(*args, **kwargs)

# Loading and fingerprint functions 
def load_dataset(file_path):
    data = pd.read_csv(file_path, sep=" ", header=None, names=["SMILES", "Name", "DockingScore"])
    return data

# to generate the smiles
def smiles_to_fp(smiles, method="morgan2", n_bits=2048):

    """
    Encode a molecule from a SMILES string into a fingerprint.

    Parameters
    ----------
    smiles : str
        The SMILES string defining the molecule.

    method : str
        The type of fingerprint to use. Default is MACCS keys.

    n_bits : int
        The length of the fingerprint.

    Returns
    -------
    array
        The fingerprint array.

    """

    # convert smiles to RDKit mol object


    fingerprints = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            if method == "morgan2":
                fp = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
                fingerprints.append(np.array(fp.GetFingerprint(mol)))

            if method == "maccs":
                fingerprints.append(np.array(MACCSkeys.GenMACCSKeys(mol)))

            if method == "morgan3":
                fp = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=n_bits)
                fingerprints.append(np.array(fp.GetFingerprint(mol)))
        else:
            fingerprints.append(np.zeros(n_bits))
            
    return np.array(fingerprints)

def evaluate_fold(train_idx, val_idx, fingerprints, docking_scores, n_estimators, max_depth, learning_rate, min_child_weight, subsample, colsample_bytree, gamma, reg_alpha, reg_lambda):
    X_train, X_val = fingerprints[train_idx], fingerprints[val_idx]
    y_train, y_val = docking_scores[train_idx], docking_scores[val_idx]
    
    model = xgb.XGBRegressor(
        n_estimators = n_estimators,
        max_depth = max_depth,
        learning_rate = learning_rate,
        min_child_weight = min_child_weight,
        subsample = subsample,
        colsample_bytree = colsample_bytree,
        gamma = gamma,
        reg_alpha = reg_alpha,
        reg_lambda = reg_lambda
    )
    
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    train_mse = mean_squared_error(y_train, y_train_pred)
    val_mse = mean_squared_error(y_val, y_val_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    
    return train_mse, val_mse, train_r2, val_r2

class ProgressBarCallback:
    def __init__(self, pbar):
        self.pbar = pbar
    
    def __call__(self, study, trial):
        self.pbar.update(1)

class XGBOptimization:
    def __init__(self, fingerprints, docking_scores, n_splits=5, n_jobs=5, n_startup_trials=10):
        self.fingerprints = fingerprints
        self.docking_scores = docking_scores
        self.kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        self.best_model = None
        self.best_score = float('inf')
        self.training_history = []
        self.n_jobs = n_jobs
        self.current_trial = 0
        self.n_startup_trials = n_startup_trials

    def objective(self, trial):
        self.current_trial += 1
        vprint(f"\n{'='*80}")
        vprint(f"Trial {self.current_trial} Starting")
        vprint(f"{'='*80}")
        
        n_estimators = trial.suggest_int('n_estimators', 100, 1000, log=True)
        max_depth = trial.suggest_int('max_depth', 3, 4) # Reduced range to avoid overfitting
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True) # To reduce overfitting
        min_child_weight = trial.suggest_int('min_child_weight', 10, 20) # increase range to try and reduce overfit
        subsample = trial.suggest_float('subsample', 0.5, 0.7) # More aggressive less overfit
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 0.7) # More aggressive less overfit
        gamma = trial.suggest_float('gamma', 0.3, 0.5) # Regularization term to reduce overfitting
        reg_alpha = trial.suggest_float('reg_alpha', 1.0, 10.0, log=True)  # L1 regularization
        reg_lambda = trial.suggest_float('reg_lambda', 1.0, 10.0, log=True)  # L2 regularization 1e-3, 100.0
        
        # Parallel processing for cross-validation
        fold_results = Parallel(n_jobs=self.n_jobs)(
            delayed(evaluate_fold)(
                train_idx, val_idx, 
                self.fingerprints, self.docking_scores,
                n_estimators, max_depth, learning_rate, min_child_weight, subsample, colsample_bytree, gamma, reg_alpha, reg_lambda
            )
            for train_idx, val_idx in self.kf.split(self.fingerprints)
        )
        
        # Unpack results
        train_mses, val_mses, train_r2s, val_r2s = zip(*fold_results)
        
        avg_train_mse = np.mean(train_mses)
        avg_val_mse = np.mean(val_mses)
        avg_train_r2 = np.mean(train_r2s)
        avg_val_r2 = np.mean(val_r2s)
        
        # Store the results for this trial
        self.training_history.append({
            'trial': self.current_trial,
            'n_estimators' : n_estimators,
            'max_depth' : max_depth,
            'learning_rate' : learning_rate,
            'min_child_weight' : min_child_weight,
            'subsample' : subsample,
            'colsample_bytree' : colsample_bytree,
            'gamma' : gamma,
            'reg_alpha' : reg_alpha,
            'reg_lambda' : reg_lambda
        })
        
        vprint(f"\nTrial {self.current_trial} Results:")
        vprint("-" * 40)
        vprint(f"Parameters:")
        vprint(f"  n_estimators = {n_estimators}")
        vprint(f"  max_depth = {max_depth}")
        vprint(f"  learning_rate = {learning_rate}")
        vprint(f"  min_child_weight = {min_child_weight}")
        vprint(f"  subsample = {subsample}")
        vprint(f"  colsample_bytree = {colsample_bytree}")
        vprint(f"  gamma = {gamma}")
        vprint(f"  reg_alpha = {reg_alpha}")
        vprint(f"  reg_lambda = {reg_lambda}")

        vprint(f"\nMetrics:")
        vprint(f"  Training MSE:     {avg_train_mse:.4f}")
        vprint(f"  Validation MSE:   {avg_val_mse:.4f}")
        vprint(f"  Training R²:      {avg_train_r2:.4f}")
        vprint(f"  Validation R²:    {avg_val_r2:.4f}")
        
        
        if self.current_trial > self.n_startup_trials and avg_val_mse < self.best_score:
            self.best_score = avg_val_mse
            self.best_params = {
                'trial': self.current_trial,
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'learning_rate': learning_rate,
                'min_child_weight': min_child_weight,
                'subsample': subsample,
                'colsample_bytree': colsample_bytree,
                'gamma' : gamma,
                'reg_alpha': reg_alpha,
                'reg_lambda': reg_lambda,                
                'train_mse': avg_train_mse,
                'val_mse': avg_val_mse,
                'train_r2': avg_train_r2,
                'val_r2': avg_val_r2
            }
            vprint(f"\n>>> New Best Trial Found! <<<")
            vprint(f"Current best validation MSE: {self.best_score:.4f}")
        elif self.current_trial <= self.n_startup_trials:
            vprint(f"\nSkipping model update: in startup phase (trial {self.current_trial}/{self.n_startup_trials})")
            
        
        return avg_val_mse



def train_model_with_optuna(fingerprints, docking_scores, r, n_trials=50, n_jobs=5):
    vprint("\nStarting Optuna optimization...")
    vprint(f"Number of trials: {n_trials}")
    vprint(f"Number of parallel jobs: {n_jobs}")
    vprint("-" * 80)
    
    optimizer = XGBOptimization(fingerprints, docking_scores, n_jobs=n_jobs, n_startup_trials=10)    
    
    study_name = f"xgb_optuna_r{r}"
    study = optuna.create_study(storage=f"sqlite:///{study_name}.sqlite3", study_name=study_name, direction='minimize')
    with tqdm(total=n_trials, desc="XGBoost Optimization Trials") as pbar:
        callback = ProgressBarCallback(pbar)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(optimizer.objective, n_trials=n_trials, n_jobs=n_jobs, callbacks=[callback])
    
    
    vprint("\n" + "="*80)
    vprint("Final Optimization Results")
    vprint("="*80)
    vprint(f"\nBest Trial (#{optimizer.best_params['trial']}):")
    vprint("-" * 40)
    vprint(f"Parameters:")
    
    vprint(f"  n_estimators:     {optimizer.best_params['n_estimators']}")
    vprint(f"  max_depth:        {optimizer.best_params['max_depth']}")
    vprint(f"  learning_rate:        {optimizer.best_params['learning_rate']}")
    vprint(f"  min_child_weight:        {optimizer.best_params['min_child_weight']}")
    vprint(f"  subsample:        {optimizer.best_params['subsample']}")
    vprint(f"  colsample_bytree:        {optimizer.best_params['colsample_bytree']}")
    vprint(f"  gamma:        {optimizer.best_params['gamma']}")
    vprint(f"  reg_alpha:        {optimizer.best_params['reg_alpha']}")
    vprint(f"  reg_lambda:        {optimizer.best_params['reg_lambda']}")

    vprint(f"\nMetrics:")
    vprint(f"  Training MSE:   {optimizer.best_params['train_mse']:.4f}")
    vprint(f"  Validation MSE: {optimizer.best_params['val_mse']:.4f}")
    vprint(f"  Training R²:    {optimizer.best_params['train_r2']:.4f}")
    vprint(f"  Validation R²:  {optimizer.best_params['val_r2']:.4f}")
    
    dff = pd.DataFrame.from_dict({f'Round {r} XGBoost': [optimizer.best_params['train_mse'], optimizer.best_params['val_mse'], optimizer.best_params['train_r2'], optimizer.best_params['val_r2']]},  orient='index', columns = ['MSE train', 'MSE val', 'R2 train', 'R2 val'])
    dff.to_csv(f'XGBoost_metrics_r{r}.csv', index = True, header = True)
    
    # Train final model with best parameters
    best_model = xgb.XGBRegressor(
        n_estimators=study.best_params['n_estimators'],
        max_depth=study.best_params['max_depth'],
        learning_rate=study.best_params['learning_rate'],
        min_child_weight=study.best_params['min_child_weight'],
        subsample=study.best_params['subsample'],
        colsample_bytree=study.best_params['colsample_bytree'],
        gamma=study.best_params['gamma'],
        reg_alpha=study.best_params['reg_alpha'],
        reg_lambda=study.best_params['reg_lambda'],

    )
    best_model.fit(fingerprints, docking_scores)
    
    return study.best_params, study  # Return ONLY the best parameters
   
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XGBoost Model Training with Optuna")
    parser.add_argument("input_file", help="Path to the input file with SMILES, Name, and DockingScore")
    parser.add_argument("external_file", help="Path to the external file with molecules for prediction")
    parser.add_argument("round", help="Current round number")
    parser.add_argument("--cpu", type=int, default=5, help="Number of CPUs to use for parallel jobs")
    parser.add_argument("--verbose", action = 'store_true', help="Enable verbose output")
    args = parser.parse_args()

    input_file = args.input_file
    external_file = args.external_file
    r = args.round
    n_jobs = args.cpu
    verbose = args.verbose


    dataset = load_dataset(input_file)
    
    smiles = dataset["SMILES"].tolist()
    docking_scores = dataset["DockingScore"].values
    
    # Convert to fingerprints
    fingerprints = smiles_to_fp(smiles)

    # Perform Optuna optimization to get best parameters
    best_params, study = train_model_with_optuna(fingerprints, docking_scores, r, n_trials=50, n_jobs=n_jobs)

    # Train the FINAL model on ALL data using the best Optuna parameters
    final_model = xgb.XGBRegressor(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        learning_rate=best_params['learning_rate'],
        min_child_weight=best_params['min_child_weight'],
        subsample=best_params['subsample'],
        colsample_bytree=best_params['colsample_bytree'],
        gamma=best_params['gamma'],
        reg_alpha=best_params['reg_alpha'],
        reg_lambda=best_params['reg_lambda']
        
    )
    
    # Print the shape of the data used for final training
    vprint("\nShape of data used for final model training:")
    vprint(f"Fingerprints shape: {fingerprints.shape}")
    vprint(f"Docking scores shape: {docking_scores.shape}")

    
    final_model.fit(fingerprints, docking_scores)  # Train on the entire dataset

# Store model information
    model_info = {
        "training_data_size": len(docking_scores),
        "fingerprint_radius": 2,  
        "fingerprint_n_bits": 2048, 
        "best_params": best_params,

    }

    # Save the model and information
    model_filename = f"xgb_model_and_info_r{r}.pkl"  
    with open(model_filename, "wb") as f:
        pickle.dump((final_model, model_info), f)

    vprint(f"\nFinal model and information saved to {model_filename}")



