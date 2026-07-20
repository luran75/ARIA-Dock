import os 
import shutil
import math
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from scipy import stats
import catboost 
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna
from optuna.samplers import TPESampler
import json



# ===============================================================================================================================
# Data Loading and Setup (Modified compared to original to use a single CSV file from https://github.com/sousouhou/BoostSF- SHAP)
# ===============================================================================================================================

# Specify the single master file containing all data
full_data_file = "data/full_dataset.csv"  # IMPORTANT: Change this to your file path
target_column = "pKd"                     # IMPORTANT: Change this to your target column name
pdb_column = "PdbID"                      # IMPORTANT: Change this to your PDB identifier column name

generateModelFolder = "catboostScoring-model-6-angstroms-cutoff-v10-optuna"
flagConductTest = True  # whether using test data to evaluate the model

# Ensure clean output directory
if os.path.exists(generateModelFolder):
    shutil.rmtree(generateModelFolder)
os.mkdir(generateModelFolder)

# Load the single full dataset
try:
    full_df = pd.read_csv(full_data_file)
except FileNotFoundError:
    print(f"Error: The file '{full_data_file}' was not found. Please check the path and filename.")
    exit()

# Separate features (X), target (y), and PDB names
X = full_df.drop(columns=[target_column, pdb_column])
y = full_df[target_column]
pdb_names = full_df[pdb_column]

# Split the data into training and testing sets
# We use a fixed random_state for reproducibility
if flagConductTest:
    Xtrain, Xtest, ytrain, ytest, pdbnametrain, pdbnametest = train_test_split(
        X, y, pdb_names, test_size=0.2, random_state=42
    )
else:
    Xtrain = X
    ytrain = y
    pdbnametrain = pdb_names
    Xtest = None
    ytest = None
    pdbnametest = None

print(f"Training data shape: {Xtrain.shape}")
if flagConductTest:
    print(f"Test data shape: {Xtest.shape}")


    # === Save PDB IDs used in each split ===
    train_pdb_file = os.path.join(generateModelFolder, "train_pdb_ids.txt")
    test_pdb_file = os.path.join(generateModelFolder, "test_pdb_ids.txt")

    with open(train_pdb_file, "w") as f:
        for pdb in pdbnametrain:
            f.write(str(pdb) + "\n")

    with open(test_pdb_file, "w") as f:
        for pdb in pdbnametest:
            f.write(str(pdb) + "\n")

    print(f"Saved train PDB IDs to {train_pdb_file}")
    print(f"Saved test PDB IDs to {test_pdb_file}")


# Define 5x2 repeated CV see line 102 (use ''' to cancel this and change line 118 back to cv=5)
rkf = RepeatedKFold(n_splits=5, n_repeats=2, random_state=42)
cv_indices = list(rkf.split(Xtrain, ytrain))




# =======================================================================================
# Optuna Objective Function (added hyperparameter tuning compared to the original script)
# =======================================================================================

def objective(trial):
    # Define hyperparameter search space - constrained to reduce overfitting
    params = {
        'iterations': trial.suggest_int('iterations', 200, 2000),  #  max iterations
        'depth': trial.suggest_int('depth', 3, 7),  #  max depth
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'random_strength': trial.suggest_float('random_strength', 0.1, 10, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 3, 50, log=True),  #  min regularization
        'rsm': trial.suggest_float('rsm', 0.5, 1.0),  #  min RSM
        'border_count': trial.suggest_categorical('border_count', [32, 64, 128]),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 50, 500),  #  min value
        'boosting_type': trial.suggest_categorical('boosting_type', ['Ordered', 'Plain']),
        'loss_function': 'RMSE',
        'random_seed': 42,
        'verbose': 100,
        'allow_writing_files': False
    }
    
    # Create and train model
    model = catboost.CatBoostRegressor(**params)
    
    # Use cross-validation to get a robust estimate
    cv_scores = cross_val_score(
        model, Xtrain, ytrain, 
        cv=cv_indices, # cv=5,  THIS GOES WILL LINE 83 TO ADD OR NOT repeated CV =============================== 
        scoring='neg_root_mean_squared_error',
        n_jobs=1  # CatBoost handles parallelization internally
    )
    
    # Return the mean RMSE (Optuna minimizes by default)
    return -cv_scores.mean()

# ==============================================================================
# Run Optuna Optimization
# ==============================================================================

print("Starting Optuna hyperparameter optimization...")

# Create study with TPE sampler for better exploration
study = optuna.create_study(
    direction='minimize',
    sampler=TPESampler(seed=42),
    study_name='catboost_optimization'
)

# Optimize with more trials for better exploration
study.optimize(
    objective, 
    n_trials=150,  # Increased from 80 for better exploration
    show_progress_bar=True,
    n_jobs=1  # CatBoost handles parallelization
)

# ======================================================================================
# Get Best Parameters and Train Final Model 1
# ======================================================================================

best_params = study.best_params
best_score = study.best_value

print(f"Best RMSE: {best_score:.4f}")
print(f"Best parameters: {best_params}")

# Save optimization results
optimization_results = {
    'best_params': best_params,
    'best_rmse': best_score,
    'n_trials': len(study.trials),
    'optimization_history': [
        {'trial': i, 'value': trial.value, 'params': trial.params} 
        for i, trial in enumerate(study.trials) if trial.value is not None
    ]
}

with open(f"{generateModelFolder}/optuna_optimization_results.json", "w") as f:
    json.dump(optimization_results, f, indent=4)

# Train best model on training data with early stopping
best_model_params = best_params.copy()
best_model_params.update({
    'loss_function': 'RMSE',
    'random_seed': 42,
    'verbose': 100,
    'early_stopping_rounds': 50,
    'use_best_model': True
})

best_model = catboost.CatBoostRegressor(**best_model_params)
# Create validation set from training data for early stopping
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    Xtrain, ytrain, test_size=0.2, random_state=42
)
best_model.fit(X_train_split, y_train_split, eval_set=(X_val_split, y_val_split))

# ==============================================================================
# Performance Reporting Functions
# ==============================================================================

def generate_performance_report(model, X, y, dataset_name):
    y_pred = model.predict(X)
    
    Rp, pvalue = stats.pearsonr(y, y_pred)
    MSE = mean_squared_error(y, y_pred)
    RMSE = np.sqrt(MSE)
    R2 = r2_score(y, y_pred)
    MAE = mean_absolute_error(y, y_pred)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y, y_pred, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.xlabel("Experimental Binding Affinity")
    plt.ylabel("Predicted Binding Affinity")
    plt.title(f"{dataset_name}: Predicted vs Experimental Binding Affinity")
    
    plt.annotate(
        f"Pearson R: {Rp:.4f}\n"
        f"RMSE: {RMSE:.4f}\n"
        f"R²: {R2:.4f}\n"
        f"MAE: {MAE:.4f}", 
        xy=(0.05, 0.95), 
        xycoords='axes fraction', 
        verticalalignment='top'
    )
    
    plt.tight_layout()
    plt.savefig(f"{generateModelFolder}/correlation-plot-{dataset_name.lower().replace(' ', '-')}.png", dpi=300)
    plt.close()
    
    return {
        'Pearson R': Rp,
        'RMSE': RMSE,
        'R2': R2,
        'MAE': MAE
    }

# Generate performance reports
train_performance = generate_performance_report(best_model, Xtrain, ytrain, "Training Data")
if flagConductTest:
    test_performance = generate_performance_report(best_model, Xtest, ytest, "Test Data")

# ==============================================================================
# Generate Optuna Optimization Plots
# ==============================================================================

# Plot optimization history
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
optuna.visualization.matplotlib.plot_optimization_history(study)
plt.title("Optimization History")

plt.subplot(1, 2, 2)
optuna.visualization.matplotlib.plot_param_importances(study)
plt.title("Parameter Importance")

plt.tight_layout()
plt.savefig(f"{generateModelFolder}/optuna_optimization_plots.png", dpi=300, bbox_inches='tight')
plt.close()

# Plot parameter relationships for top parameters
important_params = list(study.best_params.keys())[:4]  # Top 4 most important
if len(important_params) >= 2:
    fig = optuna.visualization.matplotlib.plot_slice(study, params=important_params[:4])
    plt.savefig(f"{generateModelFolder}/optuna_parameter_slice.png", dpi=300, bbox_inches='tight')
    plt.close()

# ==============================================================================
# Model Information Report
# ==============================================================================

strMI = f"""\
* Optuna Hyperparameter Optimization Results:
Best RMSE (CV): {best_score:.4f}
Number of trials: {len(study.trials)}
Best Parameters: {best_params}

* Training Data:
File used: {full_data_file}
Number of samples: {Xtrain.shape[0]}
Number of features: {Xtrain.shape[1]}

* Training Data Performance:
Pearson R: {train_performance['Pearson R']:.4f}
RMSE: {train_performance['RMSE']:.4f}
R²: {train_performance['R2']:.4f}
MAE: {train_performance['MAE']:.4f}
"""

if flagConductTest:
    strMI += f"""\
* Test Data:
File used: {full_data_file}
Number of samples: {Xtest.shape[0]}
Number of features: {Xtest.shape[1]}

* Test Data Performance:
Pearson R: {test_performance['Pearson R']:.4f}
RMSE: {test_performance['RMSE']:.4f}
R²: {test_performance['R2']:.4f}
MAE: {test_performance['MAE']:.4f}
"""

with open(f"{generateModelFolder}/model-information.txt", "w") as pf:
    pf.write(strMI)

# ==============================================================================
# SHAP Figure Generation
# ==============================================================================
print("Generating SHAP plots...")
explainer = shap.TreeExplainer(best_model)
shap_values = explainer(Xtrain)

plt.figure(figsize=(12, 8))
shap.plots.beeswarm(shap_values, max_display=40, show=False)
plt.title("SHAP Beeswarm Plot - Feature Importance")
plt.tight_layout()
plt.savefig(f"{generateModelFolder}/SHAP-beeswarm.png", dpi=600)
plt.close()

shap.summary_plot(shap_values, Xtrain, max_display=60, plot_type="bar", show=False)
fig01 = plt.gcf()
ax01 = plt.gca()
ax01.set_title("global importance of each feature for the training data\n\n Note: Feature C/C represents C(in receptor)/C(in ligand).")
ax01.set_ylabel("Features", fontsize=14)
fig01.savefig(f"{generateModelFolder}/SHAP-summary_plot.png", dpi=600, bbox_inches='tight')
plt.close()

print("Hyperparameter optimization and model training completed successfully!")

# ========================================================================================
# Train Final Model (model 2) on Full Dataset (step added compared to the original script)
# ========================================================================================
print("Training final model on full dataset...")

# Train final model with best hyperparameters on all data with early stopping
final_params = best_params.copy()
final_params.update({
    'loss_function': 'RMSE',
    'random_seed': 42,
    'verbose': 100,
    'early_stopping_rounds': 50,
    'use_best_model': True
})

full_model = catboost.CatBoostRegressor(**final_params)
# Create validation set from full dataset for early stopping
X_full_train, X_full_val, y_full_train, y_full_val = train_test_split(
    X, y, test_size=0.15, random_state=42
)
full_model.fit(X_full_train, y_full_train, eval_set=(X_full_val, y_full_val))

# Save the final model
full_model.save_model(f"{generateModelFolder}/full-final-model-catBoost-trained-all-data-optuna.cbm", format="cbm")

# Generate performance report for full model - here it is a Training Performance report (not test set at this stage) #############################
# The resulting correlation plot will show training performance only.
full_performance = generate_performance_report(full_model, X, y, "Full Dataset")

final_model_info = f"""\n--------------------\n
* The final model is trained on the full dataset:
File used: {full_data_file}

The total number of samples used: {X.shape[0]}.
The number of features: {X.shape[1]}.

* Full Dataset Performance:
Pearson R: {full_performance['Pearson R']:.4f}
RMSE: {full_performance['RMSE']:.4f}
R²: {full_performance['R2']:.4f}
MAE: {full_performance['MAE']:.4f}

* Best Hyperparameters Used:
{json.dumps(best_params, indent=2)}
"""

with open(f"{generateModelFolder}/final_model_info.txt", "w") as pf:
    pf.write(final_model_info)  
    
print("FINAL model on ALL data completed successfully!")
print(f"All results saved in: {generateModelFolder}/")
