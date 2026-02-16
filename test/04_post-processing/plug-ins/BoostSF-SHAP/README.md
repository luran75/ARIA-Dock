# BoostSF-SHAP
- **Current version = 0.1.2**

### Overview

A gradient boosting-based software for protein-ligand binding affinity prediction with SHAP explanations.

- GBDT model (chorsen from Catboost, XGBoost, LightGBM) are used to construct the scoring function.

- The global SHAP explanation and model analysis are provided for a trained model.

- Predict the binding affinity for a protein-ligand complex (with local SHAP explanation) or a set of protein-ligand complexes.
 


### Dependencies and Installation

- NumPy (>=1.26.1)
- scikit-learn (>=1.5.0)
- pandas (>=2.1.2)
- SciPy (>=1.11.3);
- Matplotlib (>=3.8.2)
- CatBoost (>=1.2.3)
- LightGBM (>=4.4.0)
- XGBoost (>=2.1.0)
- SHAP (>=0.46.0 , important!)
- AutoDockTools (for formatting a ligand and a protein structure files as PDBQT files, if necessary. downloaded URL: https://ccsb.scripps.edu/download/554/ )

After downloading the code, use pip to install Dependencies, such as:
```
pip install numpy==1.26.1
```


### Usage

#### 1. Train model, obtain the global SHAP explanation and model analysis

Train a CatBoost (or XGBoost, LightGBM) model by using the **models/train-catboost-SHAP.py** program. 
Set the parameters in train-catboost-SHAP.py : 
```
Xtrainfile = "../data/pdbbind-2020refined-exclude-2016core-x41features.csv"
ytrainfile = "../data/pdbbind-2020refined-exclude-2016core-y.txt"
pdbnametrain = "../data/pdbbind-2020refined-exclude-2016core-pdbnamelist.txt"

generateModelFolder  = "catboost-model01"  # the folder to store result

flagConductTest = True  # whether using test data to evaluate the model
if flagConductTest == True :
    Xtestfile = "../data/pdbbind-2016core-x41features.csv"
    ytestfile = "../data/pdbbind-2016core-y.txt"
    pdbnametest = "../data/pdbbind-2016core-pdbnamelist.txt"   

Cat_params = {   # parameters for catboost
    'iterations': 1000, 
    'depth': 8,
    'learning_rate': 0.02,
    'random_seed': 0
}
```

Then, type the following commands at Windows PowerShell:
```
PS D:\BoostSF-SHAP> cd models
PS D:\BoostSF-SHAP\models> python train-catboost-SHAP.py
```

Results are stored in the folder **catboost-model01**.



#### 2. Predict the binding affinity for a protein-ligand complex (with local SHAP explanation)

Type the following command for help.
```
PS D:\BoostSF-SHAP> python BoostSF-SHAP-main.py -h
usage: python BoostSF-SHAP-main.py [-h] --modeltype
                                   {catboost,lightgbm,xgboost,sklearn} --model
                                   MODEL [--receptor RECEPTOR]
                                   [--ligand LIGAND] [--rllist RLLIST]

This program is used for protein-ligand binding affinity prediction. The local
SHAP explanation can be provided.

options:
  -h, --help            show this help message and exit
  --modeltype {catboost,lightgbm,xgboost,sklearn}
                        The type of trained model.
  --model MODEL         The path of the trained model file.
  --receptor RECEPTOR   The path of the receptor file, PDBQT file format.
  --ligand LIGAND       The path of the ligand file, PDBQT file format.
  --rllist RLLIST       The path of the list file, which contains the pathes
                        of receptor and ligand files.
```

For example, predict 1a30 protein-ligand complex: 
```
PS D:\BoostSF-SHAP> python BoostSF-SHAP-main.py --modeltype catboost --model ./models/catboost-model01/trained-model.cbm --receptor ./test/1a30_protein.pdbqt --ligand ./test/1a30_ligand.pdbqt

 * The predicted binding affinity is : 5.8629

 * See { ./result-20240731-104514 } for more information.
```
Results are stored in the result folder **result-timestamp**.




#### 3. Predict the binding affinity for a set of protein-ligand complexes

Prepare a text file **list01.txt** :
```
./test/1a30_protein.pdbqt; ./test/1a30_ligand.pdbqt
./test/1bcu_protein.pdbqt; ./test/1bcu_ligand.pdbqt
./test/1c5z_protein.pdbqt; ./test/1c5z_ligand.pdbqt
```

Then, type the following command:
```
PS D:\BoostSF-SHAP> python BoostSF-SHAP-main.py --modeltype catboost --model ./models/catboost-model01/trained-model.cbm --rllist ./test/list01.txt

 * The predicted binding affinities are : [5.86294745 4.19860563 4.48173005]

 * See { ./result-20240731-104701-list } for more information.
```

Results are stored in the folder **result-timestamp-list**.


#### 4. Utilities

To format a protein and a ligand structure files as PDBQT files, types the following command for handling the 1a30 protein-ligand complex: 
```
PS D:\BoostSF-SHAP\utilities> python prepare-pdbqt-file.py --receptor ./1a30/1a30_protein.pdb --ligand ./1a30/1a30_ligand.mol2  --outfolder ./1a30
```


To extract features from a set of protein-ligand complexes and generate tabular data, prepare a text file **receptor-ligand-list.txt**. Then extract features from this set of protein-ligand complexes:
```
PS D:\BoostSF-SHAP\utilities> python prepare-features-file.py --rllist receptor-ligand-list.txt
```




### Revision records

- 0.1.2
Modify the calculation of SD.

- 0.1.1
Updating README.md. 

- 0.1.0
Releasing the software. 



### Citations
X. Chen, S. Song, Z. Song, S. Song, J. Ji, BoostSF-SHAP: Gradient boosting-based software for protein–ligand binding affinity prediction with explanations, Neurocomputing 622 (2025) 129303.
https://www.sciencedirect.com/science/article/abs/pii/S0925231224020745  


### LICENSE
BoostSF-SHAP is available under Apache License.








