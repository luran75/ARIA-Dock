# ARIA-Dock
- **Current version = 1.0**
 
### Overview
 
An active-learning based docking workflow to screen large libraries of molecules on a selected target and extract well performing compounds
 
ARIA-Dock (ActiveLearning Regression Iterative Approach to Docking) is a novel user-friendly active learning-powered workflow, for large scale, docking-based active learning screening. It allows for fast and effective hit identification through a combination of multiple machine learning methods by screening a user-defined molecular library.
 
 
### Installation
 
To use all the functions that ARIA-Dock has to offer, download the directory, open a terminal in the directory, and create a new conda environment with the provided yml file.
 
`conda env create -n aria --file aria_universal.yml`
 
`conda activate aria`
 
### Usage
 
#### Data Preparation
 
Aria-Dock takes as input a protein target (in a pdb format), a ligand to use as reference for molecular docking (pdb format), and a chemical library to screen (smi/txt format).
The molecular library needs to have smiles first followed by the names of the respective molecules. The separation can be made by either a tab or a white space. No header is needed. An example is provided below.
 
```
CS(=O)(=O)NCCC1=CC=C(S1)C2=CSC(COC=3C=CC=CC3)=N2 Z227977912
COC=1C=CC(=CC1)S(=O)(=O)N2CCCC(C2)C(=O)N(CC=3C=CC=CC3)C=4C=CC=CN4 Z228205928
CC(=O)NC=1C=CC(=CC1Cl)NC(=O)C2CCCN2C3=NC=NC4=CC=CC=C34 Z228269080
O=C(NCC1CN(CCO1)C2CC2)C3=CC=4C=CC(=CC4N3)OC(F)(F)F Z1234926942
```


#### Screening
 
Type the following command to see the how to run the workflow: `python aria.py -h` 
```
usage: aria-dock [-h] -p PROTEIN -l LIGAND -d DATABASE [-r ROUNDS]
                 [-m [{auto,rf,ri,xgb,svr,knr} ...]] [-n NUMBER] [--version]
                 [--log] [--cpu CPU] [--docking_chunks DOCKING_CHUNKS]
                 [--percentage PERCENTAGE] [--plug-in [{boostsf-shap} ...]]
                 [-v]
 
Active learning platform for structure-based drug design
 
options:
  -h, --help            show this help message and exit
  -p PROTEIN, --protein PROTEIN
                        Target protein file in pdb format
  -l LIGAND, --ligand LIGAND
                        Reference ligand/fragments file in pdb format
  -d DATABASE, --database DATABASE
                        Input database in .smi format
  -r ROUNDS, --rounds ROUNDS
                        Number of rounds to perform
  -m [{auto,rf,ri,xgb,svr,knr} ...], --model [{auto,rf,ri,xgb,svr,knr} ...]
                        Type of ML models to run. rf: RandomForest, ri: Ridge, xgb: XGBoost, svr:
                        Support Vector Regression, knr: K-Nearest Regressor, auto: automatic selection
                        based on metrics. Combination of multiple models is allowed: -e.g., -m rf ri xb
  -n NUMBER, --number NUMBER
                        If auto is selected, then specify the number of models to use (from 1 to 5).
  --version             show program's version number and exit
  --log                 Enables logging of the workflow progress and results
  --cpu CPU             Number CPUs to expoit dursing the molecular docking step. Default: 8
  --docking_chunks DOCKING_CHUNKS
                        Number of chunks to split the docking job into for parallelization. Default: 4
  --percentage PERCENTAGE
                        Percentage of molecules to select for the first docking round. Default: 1
  --plug-in [{boostsf-shap} ...]
                        Specify a plug-in to use with Ariadne: BoostSF-SHAP for post-processing
                        analysis
  -v, --verbose         Enable verbose output with detailed progress information

Villoutreix group
```
1. `-p --protein`
    The protein target to do the molecular docking on. It needs to be in a .pdb format
2. `-l --ligand`
    The reference ligand on the surface of the protein to use as reference for the molecular docking. It needs to be in a .pdb format
3. `-d --database`
    The database containing the molecules to screen on the protein target. For the structure of the file check the information above.
4. `-m --model`
    The machine learning models to use and combine to predict the binding affinity. rf: RandomForest, ri: Ridge, xb: XGBoost, svr: Support Vector Regression, knr: K-Nearest Regressor. Use the provided sequences of letters to combine the different models (consensus scoring). Select 'auto' to let the workflow decide which methods to use based on their performance on the provided data. WARNING: if you use 'auto' you need to specify how many methods you want to combine with the flag '--number'
5. `-n --number`
    If you use '-m auto' select the number of methods that the workflow should combine.
6. `--version`
    Run this command to check ARIA-Dock's current version.
7. `--log`
    Add this flag to log every output into a file called `aria_log.log`
8. `--cpu`
    The number of CPU cores to use for the workflow.
9. `--docking_chunks`
    Number of chunks to divide the `database` for molecular docking. The screening of the chunks will be performed in parallel to increase efficiency.
10. `--percentage`
    Percentage of the `database` to select for the initial molecular docking. N.B. If the number of molecules in the `database` is lower than 10000, the percentage is automatically set ot 1%
11. `--plug-in`
    Select the plug-ins to run at the end of the workflow. BoostSF-Shap takes the first two best performing molecules and runs an interpretability analysis generating a waterfall graph.
12. `-v --verbose`
    Call it to enable verbose mode. This will show a larger amount of lines in the terminal to make debugging easier.
 
**Example usage**
 
In the `test` folder, run the following command to test the workflow on a small scale (~200 compounds).
```
cd test/

python aria.py -p 2yrq_boxB_4.pdb -l 2yrq_boxB_4_fragments.pdb -d small_test.smi -r 5 -m rfrixb --cpu 10 --docking_chunks 4 --percentage 10 --plug-in boostsf-shap
```
The results of the `test` can already be seen in the `example` folder.


### Data analysis
 
The sdf file containing the molecules found by the workflow can be found in the original folder: `all_docked_molecules_{r}_rounds.sdf` where {r} is the number of rounds defined by the user.
The results are saved in the `06_data` folder:
* `00_Molecule_Selection` the selected molecules for the starting round.
* `01_ML` the results for all the ML models defined by the user for each round.
* `02_Docking` the results from the docking per round, including the one for the initial selection of molecules.
* `03_Data_Analysis` the `metrics` for every ML model chosen, the consensus scoring per round, and `data_viz`, which contains the `mean_variation`, how the mean of the distribution changes according to the rounds, `mean_distribution`, the distribution of the top molecules  per round, `bigger_bins_distribution`, a more detailed look at the distribution of the last round of the workflow, `boxplot`, the variation of the distribution per round in a boxplot figure.
* `04_postprocessing` the molecules selected according to a selection function* in a sdf file and their annexed html dashboard that facilitates the visualization of those compounds and their characteristics. The `plug-ins` directory contains the results of the desired plug-ins: `BoostSF-Shap` contains the `results_first_mol` and `results_second_mol` which show the waterfall graph and predicted results to help interpretability.
* `05_sdf_files` the sdf files of the molecules selected in every round.


 
 


