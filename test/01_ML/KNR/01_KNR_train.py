import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit.Chem import AllChem
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from rdkit import Chem
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator
import pickle
from sklearn.model_selection import KFold
from sklearn.model_selection import learning_curve
from rdkit import RDLogger
import optuna
from joblib import Parallel, delayed
from optuna.visualization import plot_optimization_history, plot_param_importances
import gc
from tqdm import tqdm


def vprint(*args, **kwargs):
	"""Print only if verbose mode is enabled"""
	if verbose:
		print(*args, **kwargs)


# to avoid rdkit warning message in the terminal
RDLogger.DisableLog('rdApp.*')

# Loading and fingerprint functions 
def load_dataset(file_path):
    data = pd.read_csv(file_path, sep='\s+', header=None, names=["SMILES", "Name", "DockingScore"])
    return data


# to generate the smiles
def smiles_to_fp(smiles, method="morgan2", n_bits=2048, radius = 2):

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
        try:  # Try converting the SMILES to mol object
            mol = Chem.MolFromSmiles(smi)
        except:  # Print the SMILES if there was an error in converting
            print(f'Invalid SMILES detected: {smi}')

        if mol:
            if method == "morgan2":
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
                fingerprints.append(np.array(fp))

            if method == "maccs":
                fingerprints.append(np.array(MACCSkeys.GenMACCSKeys(mol)))

            if method == "morgan3":
                fp = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=n_bits)
                fingerprints.append(np.array(fp.GetFingerprint(mol)))
        else:
            fingerprints.append(np.zeros(n_bits))
            
    return np.array(fingerprints)

def evaluate_fold(train_idx, val_idx, fingerprints, docking_scores, n_neighbors, weights, metric):
    X_train, X_val = fingerprints[train_idx], fingerprints[val_idx]
    y_train, y_val = docking_scores[train_idx], docking_scores[val_idx]
    
    model = KNeighborsRegressor(
        n_neighbors=n_neighbors,
        weights=weights,
        metric=metric
    )
    
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    train_mse = mean_squared_error(y_train, y_train_pred)
    val_mse = mean_squared_error(y_val, y_val_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    
    return train_mse, val_mse, train_r2, val_r2

    
def calculate_error(predicted_scores):
    comparing_models = predicted_scores
    comparing_models['dockscore'] = comparing_models['dockscore'].astype(float)
    comparing_models['ridge_predictions_difference'] = abs(comparing_models['dockscore']) - abs(comparing_models['ridge_predictions'])
    comparing_models['dev_KNR_predictions_difference'] = abs(comparing_models['dockscore']) - abs(comparing_models['dev_KNR_predictions'])
    comparing_models['grid_KNR_predictions_difference'] = abs(comparing_models['dockscore']) - abs(comparing_models['grid_KNR_predictions'])

    vprint(f'Total error KNR: {sum(comparing_models["ridge_predictions_difference"])}, avg: {sum(comparing_models["ridge_predictions_difference"])/len(comparing_models["dockscore"]):.2f} \n'
    f'Total error developed KNR: {sum(comparing_models["dev_KNR_predictions_difference"])}, avg: {sum(comparing_models["dev_KNR_predictions_difference"])/len(comparing_models["dockscore"]):.2f} \n'
    f'Total error grid-optimized KNR: {sum(comparing_models["grid_KNR_predictions_difference"])}, avg: {sum(comparing_models["grid_KNR_predictions_difference"])/len(comparing_models["dockscore"]):.2f}')   

    
class ProgressBarCallback:
    def __init__(self, pbar):
        self.pbar = pbar
    
    def __call__(self, study, trial):
        self.pbar.update(1)  



class KNROptimization:
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

        # Calculate the max number of neighbors based on training set size
        total_samples = len(self.docking_scores)
        min_train_size = int(total_samples * (self.kf.n_splits - 1) / self.kf.n_splits)
        max_n_neighbors = min(50, min_train_size - 1)  # Leave at least 1 sample out    
        
        n_neighbors = trial.suggest_int("n_neighbors", 1, max_n_neighbors)
        weights = trial.suggest_categorical("weights", ["uniform", "distance"])
        metric = trial.suggest_categorical("metric", ["euclidean", "manhattan", "hamming"])
        
        # Parallel processing for cross-validation
        kf_splits = list(self.kf.split(self.fingerprints))  # Store splits in a list
        gc.collect()  # Clean up memory before trials
        fold_results = Parallel(n_jobs=self.n_jobs)(
            delayed(evaluate_fold)(
                train_idx, val_idx, 
                self.fingerprints, self.docking_scores,
                n_neighbors, weights, metric
            )
            for train_idx, val_idx in kf_splits
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
            'n_neighbors': n_neighbors,
            'weights': weights,
            'metric': metric,
            'train_mse': avg_train_mse,
            'val_mse': avg_val_mse,
            'train_r2': avg_train_r2,
            'val_r2': avg_val_r2
        })
        
        vprint(f"\nTrial {self.current_trial} Results:")
        vprint("-" * 40)
        vprint(f"Parameters:")
        vprint(f"  n_neighbors: {n_neighbors}")
        vprint(f"  weights: {weights}")
        vprint(f"  metric: {metric}")
        vprint(f"\nMetrics:")
        vprint(f"  Training MSE:     {avg_train_mse:.4f}")
        vprint(f"  Validation MSE:   {avg_val_mse:.4f}")
        vprint(f"  Training R²:      {avg_train_r2:.4f}")
        vprint(f"  Validation R²:    {avg_val_r2:.4f}")
        
        if self.current_trial > self.n_startup_trials and avg_val_mse < self.best_score:
            self.best_score = avg_val_mse
            self.best_params = {
                'trial': self.current_trial,
                'n_neighbors': n_neighbors,
                'weights': weights,
                'metric': metric,
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

    optimizer = KNROptimization(fingerprints, docking_scores, n_jobs=n_jobs, n_startup_trials=10)
    study_name=f"knr_optuna_r{r}"
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
    vprint(f"  n_neighbors: {optimizer.best_params['n_neighbors']}")
    vprint(f"  weights: {optimizer.best_params['weights']}")
    vprint(f"  metric: {optimizer.best_params['metric']}")
    vprint(f"\nMetrics:")
    vprint(f"  Training MSE:   {optimizer.best_params['train_mse']:.4f}")
    vprint(f"  Validation MSE: {optimizer.best_params['val_mse']:.4f}")
    vprint(f"  Training R²:    {optimizer.best_params['train_r2']:.4f}")
    vprint(f"  Validation R²:  {optimizer.best_params['val_r2']:.4f}")
    
    dff = pd.DataFrame.from_dict({f'Round {r} KNR': [optimizer.best_params['train_mse'], optimizer.best_params['val_mse'], optimizer.best_params['train_r2'], optimizer.best_params['val_r2']]},  orient='index', columns = ['MSE train', 'MSE val', 'R2 train', 'R2 val'])
    dff.to_csv(f'KNR_metrics_r{r}.csv', index = True, header = True)
    
    # Train final model with best parameters
    best_model = KNeighborsRegressor(
        n_neighbors=study.best_params['n_neighbors'],
        weights=study.best_params['weights'],
        metric=study.best_params['metric'],
    )
    best_model.fit(fingerprints, docking_scores)
    
    # generate the learning curve
    train_sizes, train_scores, valid_scores = learning_curve(
    estimator=best_model,
    X=fingerprints,
    y=docking_scores,
    cv=5,
    scoring='neg_mean_squared_error',
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=n_jobs,
    shuffle=True,
    random_state=42)
    
    # plot the learning curve
    train_mean = -np.mean(train_scores, axis=1)
    valid_mean = -np.mean(valid_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training error', marker='o')
    plt.plot(train_sizes, valid_mean, label='Validation error', marker='s')

    plt.xlabel('Training Set Size')
    plt.ylabel('MSE')
    plt.title(f'Learning Curve Round {r} - Random Forest')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'learning_curve_r{r}.png')
    
    
    # Save optimization history to CSV
    history_df = pd.DataFrame(optimizer.training_history)
    history_df.to_csv(f"KNR_optimization_history_r{r}.csv", index=False)
    vprint(f"\nOptimization history saved to 'KNR_optimization_history_r{r}.csv'")
    
    optimization_history = plot_optimization_history(study)
    param_importance = plot_param_importances(study)

    optimization_history.write_image(f"optuna_optimization_history_r{r}_KNR-regression.jpg", format="jpg")
    param_importance.write_image(f"optuna_param_importance_KNR-regression_r{r}.jpg", format="jpg")
    
    #return best_model, study
    return study.best_params, study  # Return ONLY the best parameters


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Random Forest Model Training with Optuna")
    parser.add_argument("input_file", help="Path to the input file with SMILES, Name, and DockingScore")
    parser.add_argument("external_file", help="Path to the external file with molecules for prediction")
    parser.add_argument("round", help="Current round number")
    parser.add_argument("--cpu", type=int, default=5, help="Number of CPUs to use for parallel jobs")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()


    verbose = args.verbose    
    input_file = args.input_file
    external_file = args.external_file
    r = args.round
    n_jobs = args.cpu

    dataset = load_dataset(input_file)
    
    smiles = dataset["SMILES"].tolist()
    docking_scores = dataset["DockingScore"].values
    
    # Convert to fingerprints
    fingerprints = smiles_to_fp(smiles)

    # Perform Optuna optimization to get best parameters
    best_params, study = train_model_with_optuna(fingerprints, docking_scores, r, n_trials=50, n_jobs=n_jobs)

    # Train the FINAL model on ALL data using the best Optuna parameters
    final_model = KNeighborsRegressor(
        n_neighbors=best_params['n_neighbors'],
        weights=best_params['weights'],
        metric=best_params['metric']
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
    model_filename = f"KNR_model_and_info_r{r}.pkl"  
    with open(model_filename, "wb") as f:
        pickle.dump((final_model, model_info), f)  # Save as a tuple

    print(f"\nFinal model and information saved to {model_filename}")



    

