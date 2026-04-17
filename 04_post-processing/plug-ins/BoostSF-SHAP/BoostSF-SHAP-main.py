import argparse
import os
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

import catboost
import lightgbm 
import xgboost
from pathlib import Path
from sklearn import tree  
import joblib

from libboostsf import calculateFeatures


# command-line arg parse
dirname, filename = os.path.split( __file__ )
parser = argparse.ArgumentParser(prog = "python %s"%filename,
                                 description = "This program is used for protein-ligand binding affinity prediction. The local SHAP explanation can be provided.")

parser.add_argument("--modeltype", required=True, choices=['catboost', 'lightgbm', 'xgboost','sklearn'], help="The type of trained model.")
parser.add_argument("--model",     required=True, help="The path of the trained model file.")
parser.add_argument("--receptor",  help="The path of the receptor file, PDBQT file format.")
parser.add_argument("--reference",    help="The path of the refrence file to carry out smina docking, PDBQT file format.")
parser.add_argument("--ligand",    help="The path of the ligand file, PDBQT file format.")
parser.add_argument("--rllist",    help="The path of the list file, which contains the paths of receptor, reference, and ligand files.")
args = parser.parse_args()


# make result folder
time1 = datetime.datetime.now()
timestr = time1.strftime('%Y%m%d-%H%M%S')
resultFolder = f"./results-{Path(args.ligand).stem}"
if args.rllist :
    resultFolder += "-list"
os.mkdir(resultFolder)


if args.receptor and args.ligand and args.reference:   # two parameter are set, then True

    # calculate features
    features41 = calculateFeatures.cal41features(args.receptor, args.ligand, args.reference)
    features41np = np.array(features41).reshape( (1,-1) )         # 1-D array to 2-D matrix
    features41pd = pd.DataFrame(features41np, columns="C/C,N/C,O/C,S/C,C/N,N/N,O/N,S/N,C/O,N/O,O/O,S/O,C/S,N/S,O/S,S/S,C/P,N/P,O/P,S/P,C/F,N/F,O/F,S/F,C/Cl,N/Cl,O/Cl,S/Cl,C/Br,N/Br,O/Br,S/Br,C/I,N/I,O/I,S/I,gauss1(inter),gauss2(inter),repulsion(inter),hydrophobic(inter),Hydrogen(inter)".split(',') )

    # load model 
    if args.modeltype.lower() == "catboost" :
        trainedmodel = catboost.CatBoostRegressor()
        trainedmodel.load_model(args.model)

    if args.modeltype.lower() == "lightgbm" :
        trainedmodel = lightgbm.Booster(model_file=args.model)
        
    if args.modeltype.lower() == "xgboost" :
        trainedmodel = xgboost.XGBRegressor()
        trainedmodel.load_model(args.model)  
    
    if args.modeltype.lower() == "sklearn" :
        trainedmodel = joblib.load(args.model)

    # predicted binding affinity
    preds = trainedmodel.predict(features41pd)
    print( "\n * The predicted binding affinity is : %.4f"%preds[0] )


    # write to a result file
    str01 = ""
    for fea, val in zip( list(features41pd.columns),  list(features41pd.iloc[0,:]) ):
        str01 += "%12.4f  # %s\n"%(val, fea)

    strresult = """\
* The used model is : {%s}
    
* The path of receptor file is : { %s }
* The path of ligand file is : { %s }

* The calculated features are :
Note: Feature C/C represents C(in receptor)/C(in ligand).
%s
* The predicted binding affinity is : %.4f
    """%(args.model, args.receptor, args.ligand, str01, preds[0] )

    pf = open("%s/predicted-result.txt"%resultFolder, "w")
    pf.write(strresult)
    pf.close()


    # SHAP value
    explainer = shap.TreeExplainer(trainedmodel)
    shap_values = explainer(features41pd)  

    shap.plots.waterfall(shap_values[0], max_display=60, show=False) # not show
    fig01 = plt.gcf()
    ax01 = plt.gca()
    ax01.set_title( "waterfall figure\n\n Note: Feature C/C represents C(in receptor)/C(in ligand)." )
    ax01.set_ylabel("Features", fontsize=14)
    fig01.savefig("%s/waterfall.png"%resultFolder, dpi=600, bbox_inches='tight')
    plt.clf()  # clear figure


    shap.plots.force(shap_values[0], matplotlib=True, show=False)  # not show
    fig02 = plt.gcf()
    ax02 = plt.gca()
    fig02.savefig("%s/force.png"%resultFolder, dpi=600, bbox_inches='tight')



if args.rllist :  #  parameter are set, then True. Else 'none' is False
    
    Xtestpd = pd.DataFrame(columns="C/C,N/C,O/C,S/C,C/N,N/N,O/N,S/N,C/O,N/O,O/O,S/O,C/S,N/S,O/S,S/S,C/P,N/P,O/P,S/P,C/F,N/F,O/F,S/F,C/Cl,N/Cl,O/Cl,S/Cl,C/Br,N/Br,O/Br,S/Br,C/I,N/I,O/I,S/I,gauss1(inter),gauss2(inter),repulsion(inter),hydrophobic(inter),Hydrogen(inter)".split(',') )
    
    pfrl = open( args.rllist, "r")
    rllistLines = pfrl.readlines()
    pfrl.close()

    verifiedLines = []
    for rlLine in rllistLines :
        if rlLine.lower().count(".pdbqt") == 3 :  # confirm three pdbqt file.
            verifiedLines.append( "  %s"%(rlLine.strip()) )
            
            fileList = rlLine.split(";")
            rFile = fileList[0].strip()  # receptor file
            lFile = fileList[1].strip()  # ligand file
            refFile = fileList[2].strip()  # reference ligand file for autobox 
            
            features41 = calculateFeatures.cal41features(rFile, lFile, refFile )
            Xtestpd.loc[ len(Xtestpd.index) ] = features41  # append as a new row.
    
    Xsave = Xtestpd.copy(deep=True)
    Xsave['receptor-ligand'] = verifiedLines  # add a comment column
    Xsave.to_csv("%s/calculated-features.csv"%resultFolder, float_format='%10.4f', index=False)
    
    # load model 
    if args.modeltype.lower() == "catboost" :
        trainedmodel = catboost.CatBoostRegressor()
        trainedmodel.load_model(args.model)

    if args.modeltype.lower() == "lightgbm" :
        trainedmodel = lightgbm.Booster(model_file=args.model)
        
    if args.modeltype.lower() == "xgboost" :
        trainedmodel = xgboost.XGBRegressor()
        trainedmodel.load_model(args.model)  

    if args.modeltype.lower() == "sklearn" :
        trainedmodel = joblib.load(args.model)

    # predicted binding affinity
    preds = trainedmodel.predict(Xtestpd)
    print("\n * The predicted binding affinities are : ", end="")
    print( preds)
    
    ytestpd = pd.DataFrame(columns=["predicted binding affinity", "receptor-ligand"] )
    ytestpd["predicted binding affinity"] = preds
    ytestpd["receptor-ligand"] = verifiedLines # add a comment column
    ytestpd.to_csv("%s/predicted-binding-affinity.csv"%resultFolder, float_format='%12.4f', index=False)
    
    pflog = open("%s/result-log.txt"%resultFolder, "w")
    str02 = "* The used model is : {%s}\n"%( args.model )
    pflog.write( str02 )
    pflog.close()

print( "\n * See { %s } for more information. \n"%resultFolder  )

