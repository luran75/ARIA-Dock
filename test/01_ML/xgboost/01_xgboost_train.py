import argparse
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rdkit import Chem, RDLogger
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator

import xgboost as xgb
from sklearn.model_selection import KFold, learning_curve
from sklearn.metrics import mean_squared_error, r2_score

import optuna
import optuna.visualization.matplotlib as optuna_mpl
from joblib import Parallel, delayed
from tqdm import tqdm

# To ignore ExperimentalWarning from Optuna when using certain features
warnings.filterwarnings('ignore', category=optuna.exceptions.ExperimentalWarning)

# to avoid rdkit warning message in the terminal
RDLogger.DisableLog('rdApp.*')

verbose = False

def vprint(*args, **kwargs):
    """Print only if verbose mode is enabled."""
    if verbose:
        print(*args, **kwargs)

# Loading and fingerprint functions
def load_dataset(file_path):
    data = pd.read_csv(file_path, sep=r'\s+', header=None, names=["SMILES", "Name", "DockingScore"])
    return data


def smiles_to_fp(smiles, method="morgan2", n_bits=2048):
    """
    Encode a list of SMILES strings into fingerprints.

    Invalid SMILES (those that RDKit cannot parse) are dropped. A boolean
    mask is returned so the caller can align the target vector to the
    valid molecules.

    Parameters
    ----------
    smiles : list of str
        The SMILES strings defining the molecules.
    method : str
        Fingerprint type: "morgan2", "morgan3", or "maccs".
    n_bits : int
        Length of the Morgan fingerprint (ignored for MACCS).

    Returns
    -------
    fingerprints : np.ndarray of shape (n_valid, n_bits or 167)
        The fingerprint matrix, only for SMILES that parsed successfully.
    valid_mask : np.ndarray of shape (n_input,), dtype bool
        True where the SMILES parsed successfully.
    """
    if method == "morgan2":
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
    elif method == "morgan3":
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=n_bits)
    elif method == "maccs":
        gen = None
    else:
        raise ValueError(f"Unknown fingerprint method: {method}")

    fingerprints = []
    valid_mask = np.zeros(len(smiles), dtype=bool)

    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is None:
            vprint(f"Invalid SMILES at index {i}, dropped: {smi!r}")
            continue

        if method == "maccs":
            fp = np.array(MACCSkeys.GenMACCSKeys(mol))
        else:
            fp = np.array(gen.GetFingerprint(mol))

        fingerprints.append(fp)
        valid_mask[i] = True

    if not fingerprints:
        raise ValueError("No valid SMILES could be parsed from the input.")

    return np.array(fingerprints), valid_mask

def evaluate_fold(train_idx, val_idx, fingerprints, docking_scores,
                  n_estimators, max_depth, learning_rate, min_child_weight,
                  subsample, colsample_bytree, gamma, reg_alpha, reg_lambda):
    X_train, X_val = fingerprints[train_idx], fingerprints[val_idx]
    y_train, y_val = docking_scores[train_idx], docking_scores[val_idx]


    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        gamma=gamma,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        objective='reg:squarederror',
        n_jobs=1,
        random_state=RANDOM_STATE,
        verbosity=0,
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
    """
    Optuna objective wrapper for XGBoost regression.

    """

    def __init__(self, fingerprints, docking_scores, n_splits=5, n_jobs=5):
        self.fingerprints = fingerprints
        self.docking_scores = docking_scores
        self.kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        self.n_jobs = n_jobs
        self.training_history = []

    def objective(self, trial):

        n_estimators = trial.suggest_int('n_estimators', 100, 1000, log=True)
        max_depth = trial.suggest_int('max_depth', 3, 10)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
        min_child_weight = trial.suggest_int('min_child_weight', 1, 10)
        subsample = trial.suggest_float('subsample', 0.5, 1.0)
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)
        gamma = trial.suggest_float('gamma', 0.0, 5.0)
        reg_alpha = trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True)
        reg_lambda = trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True)

        fold_results = Parallel(n_jobs=self.n_jobs)(
            delayed(evaluate_fold)(
                train_idx, val_idx,
                self.fingerprints, self.docking_scores,
                n_estimators, max_depth, learning_rate, min_child_weight,
                subsample, colsample_bytree, gamma, reg_alpha, reg_lambda,
            )
            for train_idx, val_idx in self.kf.split(self.fingerprints)
        )

        train_mses, val_mses, train_r2s, val_r2s = zip(*fold_results)
        avg_train_mse = float(np.mean(train_mses))
        avg_val_mse = float(np.mean(val_mses))
        avg_train_r2 = float(np.mean(train_r2s))
        avg_val_r2 = float(np.mean(val_r2s))

        trial.set_user_attr('train_mse', avg_train_mse)
        trial.set_user_attr('val_mse', avg_val_mse)
        trial.set_user_attr('train_r2', avg_train_r2)
        trial.set_user_attr('val_r2', avg_val_r2)

        self.training_history.append({
            'trial': trial.number,
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'min_child_weight': min_child_weight,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'gamma': gamma,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'train_mse': avg_train_mse,
            'val_mse': avg_val_mse,
            'train_r2': avg_train_r2,
            'val_r2': avg_val_r2,
        })

        vprint(f"\nTrial {trial.number} | n_est={n_estimators} depth={max_depth} "
               f"lr={learning_rate:.4f} | "
               f"train MSE={avg_train_mse:.4f} val MSE={avg_val_mse:.4f} "
               f"| train R²={avg_train_r2:.4f} val R²={avg_val_r2:.4f}")

        return avg_val_mse



def train_model_with_optuna(fingerprints, docking_scores, r, n_trials=50, n_jobs=5):
    vprint("\nStarting Optuna optimization...")
    vprint(f"Number of trials: {n_trials}")
    vprint(f"Number of parallel jobs (CV folds): {n_jobs}")
    vprint("-" * 80)

    optimizer = XGBOptimization(fingerprints, docking_scores, n_jobs=n_jobs)
    study_name = f"xgb_optuna_r{r}"
    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(study_name=study_name, direction='minimize', sampler=sampler)

    with tqdm(total=n_trials, desc=f"XGBoost Optuna trials (round {r})") as pbar:
        callback = ProgressBarCallback(pbar)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(optimizer.objective, n_trials=n_trials, n_jobs=1, callbacks=[callback])

    best_trial = study.best_trial
    best_params = dict(best_trial.params)
    best_metrics = dict(best_trial.user_attrs)

    vprint("\n" + "=" * 80)
    vprint("Final Optimization Results")
    vprint("=" * 80)
    vprint(f"\nBest Trial (#{best_trial.number}):")
    for k, v in best_params.items():
        if isinstance(v, float):
            vprint(f"  {k}: {v:.4f}")
        else:
            vprint(f"  {k}: {v}")
    vprint(f"  Training MSE:   {best_metrics['train_mse']:.4f}")
    vprint(f"  Validation MSE: {best_metrics['val_mse']:.4f}")
    vprint(f"  Training R²:    {best_metrics['train_r2']:.4f}")
    vprint(f"  Validation R²:  {best_metrics['val_r2']:.4f}")

    # Save metrics that correspond to the saved model
    metrics_df = pd.DataFrame.from_dict(
        {f'Round {r} XGBoost': [
            best_metrics['train_mse'],
            best_metrics['val_mse'],
            best_metrics['train_r2'],
            best_metrics['val_r2'],
        ]},
        orient='index',
        columns=['MSE train', 'MSE val', 'R2 train', 'R2 val'],
    )
    metrics_df.to_csv(f'XGBoost_metrics_r{r}.csv', index=True, header=True)

    # Best model on full training data, using the best hyperparameters
    best_model = xgb.XGBRegressor(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        learning_rate=best_params['learning_rate'],
        min_child_weight=best_params['min_child_weight'],
        subsample=best_params['subsample'],
        colsample_bytree=best_params['colsample_bytree'],
        gamma=best_params['gamma'],
        reg_alpha=best_params['reg_alpha'],
        reg_lambda=best_params['reg_lambda'],
        objective='reg:squarederror',
        n_jobs=1,
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    best_model.fit(fingerprints, docking_scores)

    # Save full optimization history
    history_df = pd.DataFrame(optimizer.training_history)
    history_df.to_csv(f"XGBoost_optimization_history_r{r}.csv", index=False)
    vprint(f"\nOptimization history saved to 'XGBoost_optimization_history_r{r}.csv'")

    # Learning curve
    train_sizes, train_scores, valid_scores = learning_curve(
        estimator=best_model,
        X=fingerprints,
        y=docking_scores,
        cv=5,
        scoring='neg_mean_squared_error',
        train_sizes=np.linspace(0.1, 1.0, 10),
        n_jobs=n_jobs,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    train_mean = -np.mean(train_scores, axis=1)
    valid_mean = -np.mean(valid_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training error', marker='o')
    plt.plot(train_sizes, valid_mean, label='Validation error', marker='s')
    plt.xlabel('Training Set Size')
    plt.ylabel('MSE')
    plt.title(f'Learning Curve Round {r} - XGBoost')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'learning_curve_r{r}.png')
    plt.close()

    # Optuna diagnostic plots
    optuna_mpl.plot_optimization_history(study)
    plt.savefig(f"optuna_optimization_history_r{r}_XGBoost-regression.jpg", format="jpg")
    plt.close()
    try:
        optuna_mpl.plot_param_importances(study)
        plt.savefig(f"optuna_param_importance_XGBoost-regression_r{r}.jpg", format="jpg")
        plt.close()
    except (ValueError, RuntimeError) as e:
        vprint(f"Skipping param-importance plot ({e}).")

    return best_params, best_metrics, study, best_model
   
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XGBoost Model Training with Optuna")
    parser.add_argument("input_file", help="Path to the input file with SMILES, Name, and DockingScore")
    parser.add_argument("round", type=int, help="Current round number")
    parser.add_argument("--cpu", type=int, default=5, help="Number of CPUs to use for parallel jobs (CV folds)")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--verbose", action='store_true', help="Enable verbose output")
    parser.add_argument("--random_state", type=int, default=42, help="Random state for reproducibility")
    args = parser.parse_args()

    input_file = args.input_file
    r = args.round
    n_jobs = args.cpu
    n_trials = args.n_trials
    verbose = args.verbose
    RANDOM_STATE = args.random_state

    dataset = load_dataset(input_file)
    smiles = dataset["SMILES"].tolist()
    docking_scores_raw = dataset["DockingScore"].values

    # Convert to fingerprints
    fingerprints, valid_mask = smiles_to_fp(smiles)
    n_dropped = (~valid_mask).sum()
    if n_dropped:
        vprint(f"Dropped {n_dropped} invalid SMILES out of {len(smiles)}.")
    docking_scores = docking_scores_raw[valid_mask]

    assert fingerprints.shape[0] == docking_scores.shape[0], (
        "Fingerprint/score length mismatch after filtering invalid SMILES."
    )

    # Optuna optimization
    best_params, best_metrics, study, final_model = train_model_with_optuna(
        fingerprints, docking_scores, r, n_trials=n_trials, n_jobs=n_jobs,
    )

    # Final model uses the SAME hyperparameters whose CV metrics were just reported
    #final_model = xgb.XGBRegressor(
    #    n_estimators=best_params['n_estimators'],
    #    max_depth=best_params['max_depth'],
    #    learning_rate=best_params['learning_rate'],
    #    min_child_weight=best_params['min_child_weight'],
    #    subsample=best_params['subsample'],
    #    colsample_bytree=best_params['colsample_bytree'],
    #    gamma=best_params['gamma'],
    #    reg_alpha=best_params['reg_alpha'],
    #    reg_lambda=best_params['reg_lambda'],
    #    objective='reg:squarederror',
    #    n_jobs=1,
    #    random_state=RANDOM_STATE,
    #    verbosity=0,
    #)

    vprint("\nShape of data used for final model training:")
    vprint(f"Fingerprints shape: {fingerprints.shape}")
    vprint(f"Docking scores shape: {docking_scores.shape}")

    #final_model.fit(fingerprints, docking_scores)

    model_info = {
        "training_data_size": len(docking_scores),
        "n_invalid_smiles_dropped": int(n_dropped),
        "fingerprint_method": "morgan2",
        "fingerprint_radius": 2,
        "fingerprint_n_bits": 2048,
        "best_params": best_params,
        "best_cv_metrics": best_metrics,
        "random_state": RANDOM_STATE,
    }

    model_filename = f"xgb_model_and_info_r{r}.pkl"
    with open(model_filename, "wb") as f:
        pickle.dump((final_model, model_info), f)

    vprint(f"\nFinal model and information saved to {model_filename}")