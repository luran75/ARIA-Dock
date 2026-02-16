
"""
Applies structural filters and calculates physicochemical properties for molecular docking results with comprehensive analysis and ranking system.
Composite score (weighted average) selects the best compounds according to the following weights attributed to each property:
Affinity Score: 0.70, minimized affinity calculated by smina which is the most important property
QED Score: 0.10, druglikeness        
SA Score: 0.00, synthetic accessibility
Lipinski Score: 0.10, compliance with Lipinski's rule of 5  
Filter Score: 0.10  structural alert compliance
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", message=".*to-Python converter for boost::shared_ptr.*")

import sys, os, argparse, json
from pathlib import Path
from typing import List, Optional
import pandas as pd, numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, QED
from rdkit.Chem import PandasTools
from rdkit.Chem import AllChem
import medchem as mc
import datamol as dm
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from rdkit import RDLogger

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

def vprint(*args, **kwargs):
	"""Print only if verbose mode is enabled"""
	if verbose:
		print(*args, **kwargs)

# ------------------ Core processing functions  ------------------

def load_molecules_from_sdf(sdf_file: str):
    """Load molecules from SDF with proper 3D coordinate handling."""
    try:
        suppl = Chem.SDMolSupplier(sdf_file, removeHs=False, sanitize=True)
        mols = []
        
        for i, mol in enumerate(suppl):
            if mol is None:
                print(f"Warning: Could not parse molecule {i}")
                continue
                
            # Ensure 3D coordinates are properly recognized
            if mol.GetNumConformers() > 0:
                conf = mol.GetConformer()
                # Check if molecule actually has 3D coordinates
                has_3d_coords = any(
                    abs(conf.GetAtomPosition(j).z) > 0.001 
                    for j in range(mol.GetNumAtoms())
                )
                
                if has_3d_coords:
                    # Force RDKit to recognize this as 3D
                    conf.Set3D(True)
                else:
                    vprint(f"Warning: Molecule {i} has no meaningful Z coordinates")
            else:
                vprint(f"Warning: Molecule {i} has no conformers")
            
            mols.append(mol)
        
        vprint(f"Successfully loaded {len(mols)} molecules with 3D coordinates")
        
        if not mols:
            raise ValueError(f"No valid molecules found in {sdf_file}")
            
        df = pd.DataFrame({
            "smiles": [dm.to_smiles(mol) for mol in mols],
            "name": [mol.GetProp("_Name") if mol.HasProp("_Name") else f"mol_{i}" for i, mol in enumerate(mols)],
            "mol": mols,
            "minimizedAffinity": [mol.GetProp('minimizedAffinity') if mol.HasProp("minimizedAffinity") else "" for mol in mols]
        })
        
        return df
    except Exception as e:
        print(f"Error loading SDF file {sdf_file}: {e}")
        raise

def calculate_sa_score_approximate(mol):
    try:
        complexity_score = 0
        ring_info = mol.GetRingInfo()
        complexity_score += len(ring_info.AtomRings()) * 0.5
        heavy_atoms = mol.GetNumHeavyAtoms()
        carbon_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C')
        heteroatom_ratio = (heavy_atoms - carbon_atoms) / heavy_atoms if heavy_atoms > 0 else 0
        complexity_score += heteroatom_ratio * 2
        complexity_score += rdMolDescriptors.CalcNumAtomStereoCenters(mol) * 0.3
        return min(10, max(1, 3 + complexity_score))
    except Exception:
        return 5.0

def calculate_physicochemical_properties(df: pd.DataFrame):
    properties = []
    for i, mol in enumerate(df['mol']):
        try:
            mw = Descriptors.MolWt(mol)
            logp = Crippen.MolLogP(mol)
            hbd = rdMolDescriptors.CalcNumHBD(mol)
            hba = rdMolDescriptors.CalcNumHBA(mol)
            tpsa = rdMolDescriptors.CalcTPSA(mol)
            rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
            aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
            heavy_atoms = mol.GetNumHeavyAtoms()
            qed_score = QED.qed(mol)
            sa_score = calculate_sa_score_approximate(mol)
            lipinski_violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
            is_lipinski_compliant = lipinski_violations == 0
            is_lead_like = (mw <= 350 and logp <= 3.5 and rotatable_bonds <= 7)
            properties.append({'MW': mw, 'LogP': logp, 'HBD': hbd, 'HBA': hba,
                               'TPSA': tpsa, 'RotBonds': rotatable_bonds, 'ArRings': aromatic_rings,
                               'HeavyAtoms': heavy_atoms, 'QED': qed_score, 'SA_Score': sa_score,
                               'Lipinski_Violations': lipinski_violations,
                               'Lipinski_Compliant': is_lipinski_compliant, 'Lead_Like': is_lead_like})
        except Exception as e:
            properties.append({'MW': np.nan, 'LogP': np.nan, 'HBD': np.nan, 'HBA': np.nan,
                               'TPSA': np.nan, 'RotBonds': np.nan, 'ArRings': np.nan,
                               'HeavyAtoms': np.nan, 'QED': np.nan, 'SA_Score': np.nan,
                               'Lipinski_Violations': np.nan, 'Lipinski_Compliant': False, 'Lead_Like': False})
    props_df = pd.DataFrame(properties)
    return pd.concat([df.reset_index(drop=True), props_df.reset_index(drop=True)], axis=1)

def get_available_alerts():
    try:
        alerts_df = mc.structural.CommonAlertsFilters.list_default_available_alerts()
        if hasattr(alerts_df, "rule_set_name"):
            return alerts_df.rule_set_name.tolist()
        if isinstance(alerts_df, (list, tuple)):
            return list(alerts_df)
        if hasattr(alerts_df, "tolist"):
            return alerts_df.tolist()
    except Exception as e:
        print(f"Error getting alerts: {e}")
        try:
            return mc.structural.CommonAlertsFilters.list_default_available_rule_sets()
        except Exception:
            pass
    return ['PAINS', 'BRENK']

def apply_structural_alerts(df: pd.DataFrame, alerts_list: List[str]):
    results_data = {"smiles": df["smiles"].tolist(), "name": df["name"].tolist(),
                    "mol": df["mol"].tolist(), "minimizedAffinity": df["minimizedAffinity"].tolist()}
    prop_columns = ['MW','LogP','HBD','HBA','TPSA','RotBonds','ArRings','HeavyAtoms','QED','SA_Score',
                   'Lipinski_Violations','Lipinski_Compliant','Lead_Like']
    for prop in prop_columns:
        if prop in df.columns:
            results_data[prop] = df[prop].tolist()
    mols = df['mol'].tolist()
    for i, alert in enumerate(alerts_list):
        try:
            print(f'Applying {alert} filters ({i+1}/{len(alerts_list)})...')
            alerts = mc.structural.CommonAlertsFilters(alerts_set=[alert])
            results = alerts(mols=mols, n_jobs=-1, progress=True, progress_leave=True, scheduler="auto")
            pass_filter = results['pass_filter'] if isinstance(results, dict) else results.pass_filter
            status = results['status'] if isinstance(results, dict) else results.status
            reasons = results['reasons'] if isinstance(results, dict) else results.reasons
            results_data[f'{alert}_pass_filter'] = list(pass_filter)
            results_data[f'{alert}_status'] = list(status)
            results_data[f'{alert}_reasons'] = list(reasons)
        except Exception as e:
            n_mols = len(mols)
            results_data[f'{alert}_pass_filter'] = [False] * n_mols
            results_data[f'{alert}_status'] = ['FILTER_ERROR'] * n_mols
            results_data[f'{alert}_reasons'] = [f"Filter failed: {str(e)}"] * n_mols
    return pd.DataFrame(results_data)

def calculate_compound_scores(df: pd.DataFrame):
    df = df.copy()
    df['Affinity_Numeric'] = pd.to_numeric(df['minimizedAffinity'], errors='coerce')
    affinity_min, affinity_max = -14, -4
    df['Affinity_Score'] = np.clip((affinity_max - df['Affinity_Numeric']) / (affinity_max - affinity_min), 0, 1).fillna(0)
    df['QED_Score'] = df['QED'].fillna(0)
    df['SA_Score_Norm'] = np.clip((10 - df['SA_Score']) / 9, 0, 1).fillna(0)
    df['Lipinski_Score'] =  np.clip(((4 - df['Lipinski_Violations'].astype(int)) / 4), 0, 1).fillna(0)    
  
    filter_columns = [col for col in df.columns if col.endswith('_pass_filter')]
    if filter_columns:
        df['Filter_Score'] = df[filter_columns].mean(axis=1)
    else:
        df['Filter_Score'] = 1.0

    # WEIGHTS: Affinity 0.70, QED 0.10, SA 0.00, Lipinski 0.10, Filters 0.10
    weights = {'Affinity_Score': 0.70, 
               'QED_Score': 0.10, 
               'SA_Score_Norm': 0.00, 
               'Lipinski_Score': 0.10,  
               'Filter_Score': 0.10}
    df['Composite_Score'] = (df['Affinity_Score']*weights['Affinity_Score'] + df['QED_Score']*weights['QED_Score'] +
                             df['SA_Score_Norm']*weights['SA_Score_Norm'] + df['Lipinski_Score']*weights['Lipinski_Score'] +
                             df['Filter_Score']*weights['Filter_Score'])
    
    df['Rank'] = df['Composite_Score'].rank(method='dense', ascending=False).astype(int)
    return df

def save_results(df: pd.DataFrame, output_file: str):
    try:
        csv_output = f'{Path(output_file).stem}.csv'
        csv_props = [c for c in df.columns if c != 'mol']
        df[csv_props].to_csv(csv_output, index=False)
        PandasTools.WriteSDF(df, output_file, idName='name', molColName='mol', properties=csv_props)
        vprint(f"Saved CSV: {csv_output} and SDF: {output_file}")
    except Exception as e:
        print(f"Failed to save results: {e}")
        raise

def filter_passing_molecules(df: pd.DataFrame):
    filter_columns = [col for col in df.columns if col.endswith('_pass_filter')]
    if not filter_columns:
        return df
    return df[df[filter_columns].all(axis=1)].copy()

def export_top_compounds(df: pd.DataFrame, n_compounds: int, output_prefix: str, suffix: str=''):
    top_compounds = df.nlargest(n_compounds, 'Composite_Score')
    filter_columns = [col for col in df.columns if col.endswith('_pass_filter')]
    export_columns = ['Rank','name','smiles','Composite_Score','Affinity_Numeric','MW','LogP','HBD','HBA','TPSA','RotBonds','QED','SA_Score','Lipinski_Compliant','Lead_Like','Lipinski_Violations']
    export_columns.extend(filter_columns)
    available = [c for c in export_columns if c in top_compounds.columns]
    export_df = top_compounds[available].copy()
    numeric_cols = export_df.select_dtypes(include=[np.number]).columns
    export_df[numeric_cols] = export_df[numeric_cols].round(3)
    csv_file = f"{output_prefix}_top_{n_compounds}{suffix}_compounds.csv"
    export_df.to_csv(csv_file, index=False)
    if 'mol' in top_compounds.columns:
        sdf_file = f"{output_prefix}_top_{n_compounds}{suffix}_compounds.sdf"
        PandasTools.WriteSDF(top_compounds, sdf_file, idName='name', molColName='mol', properties=available)
    vprint(f"Exported top {n_compounds}{(' ' + suffix) if suffix else ''} compounds to {csv_file}")
    return csv_file

# ------------------ Visualization and dashboard building ------------------

def create_comprehensive_visualizations(df: pd.DataFrame, output_prefix: str, protein_pdb: Optional[str]=None):
    create_property_distributions(df, f"{output_prefix}_property_distributions.png")
    create_correlation_matrix(df, f"{output_prefix}_correlation_matrix.png")
    create_3d_interactive_plot(df, f"{output_prefix}_3d_analysis.html")
    create_lipinski_heatmap(df, f"{output_prefix}_lipinski_analysis.png")
    create_filter_summary(df, f"{output_prefix}_filters_summary.png")
    create_top_compounds_dashboard(df, f"{output_prefix}_top_compounds.html", protein_pdb=protein_pdb)

def create_property_distributions(df: pd.DataFrame, output_file: str):
    filter_columns = [col for col in df.columns if col.endswith('_pass_filter')]
    all_pass = df[filter_columns].all(axis=1) if filter_columns else pd.Series([True]*len(df))
    df_passing = df[all_pass]
    properties = ['MW','LogP','TPSA','QED','Affinity_Numeric']
    fig, axes = plt.subplots(2,3,figsize=(18,12))
    axes = axes.flatten()
    for i, prop in enumerate(properties):
        if prop in df.columns:
            axes[i].hist(df[prop].dropna(), bins=50, alpha=0.6, label='All molecules', density=True)
            if len(df_passing) > 0:
                axes[i].hist(df_passing[prop].dropna(), bins=50, alpha=0.8, label='Filter-passing', density=True)
            axes[i].set_xlabel(prop); axes[i].set_ylabel('Density'); axes[i].set_title(f'{prop} Distribution'); axes[i].legend(); axes[i].grid(True,alpha=0.3)
    try: fig.delaxes(axes[5])
    except: pass
    plt.tight_layout(); plt.savefig(output_file, dpi=300, bbox_inches='tight'); plt.close()
    vprint(f"Saved {output_file}")

def create_correlation_matrix(df: pd.DataFrame, output_file: str):
    numeric_props = ['MW','LogP','HBD','HBA','TPSA','RotBonds','ArRings','QED','SA_Score','Affinity_Numeric','Composite_Score']
    correlation_df = df[numeric_props].select_dtypes(include=[np.number])
    correlation_matrix = correlation_df.corr()
    plt.figure(figsize=(12,10))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0, square=True, linewidths=0.1, cbar_kws={"shrink": .5}, fmt='.2f')
    plt.title('Property Correlation Matrix', fontsize=16, pad=20)
    plt.tight_layout(); plt.savefig(output_file, dpi=300, bbox_inches='tight'); plt.close()
    vprint(f"Saved {output_file}")

def create_3d_interactive_plot(df: pd.DataFrame, output_file: str):
    df_plot = df.sample(n=2000, random_state=42) if len(df) > 2000 else df
    fig = go.Figure(data=go.Scatter3d(x=df_plot['MW'], y=df_plot['LogP'], z=df_plot['TPSA'], mode='markers',
                                     marker=dict(size=5, color=df_plot['Affinity_Numeric'], colorscale='Viridis', colorbar=dict(title="Binding Affinity"), opacity=0.7),
                                     text=df_plot['name'], hovertemplate='<b>%{text}</b><br>MW: %{x:.1f}<br>LogP: %{y:.2f}<br>TPSA: %{z:.1f}<br>Affinity: %{marker.color:.2f}<extra></extra>'))
    fig.update_layout(title='3D Property Space Analysis', scene=dict(xaxis_title='Molecular Weight', yaxis_title='LogP', zaxis_title='TPSA'), width=900, height=700)
    fig.write_html(output_file)
    vprint(f"Saved {output_file}")

def create_lipinski_heatmap(df: pd.DataFrame, output_file: str):
    violation_data = []
    for _, row in df.iterrows():
        violations = {'MW > 500': row['MW'] > 500 if not pd.isna(row['MW']) else False,
                      'LogP > 5': row['LogP'] > 5 if not pd.isna(row['LogP']) else False,
                      'HBD > 5': row['HBD'] > 5 if not pd.isna(row['HBD']) else False,
                      'HBA > 10': row['HBA'] > 10 if not pd.isna(row['HBA']) else False}
        violation_data.append(violations)
    violation_df = pd.DataFrame(violation_data)
    violation_summary = violation_df.mean() * 100
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(16,6))
    if len(df) > 100:
        sample_indices = np.random.choice(len(df), 100, replace=False)
        heatmap_data = violation_df.iloc[sample_indices]
        ax1.set_title(f'Lipinski Violations Heatmap (Random Sample of 100/{len(df)} molecules)')
    else:
        heatmap_data = violation_df
        ax1.set_title('Lipinski Violations Heatmap (All molecules)')
    sns.heatmap(heatmap_data.T, cmap='Reds', cbar_kws={'label': 'Violation'}, ax=ax1)
    ax1.set_xlabel('Molecules'); ax1.set_ylabel('Lipinski Rules')
    violation_summary.plot(kind='bar', ax=ax2, color='coral'); ax2.set_title('Lipinski Rule Violations (%)'); ax2.set_ylabel('Percentage of Molecules'); ax2.tick_params(axis='x', rotation=45); ax2.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(output_file, dpi=300, bbox_inches='tight'); plt.close()
    vprint(f"Saved {output_file}")

def create_filter_summary(df: pd.DataFrame, output_file: str):
    filter_cols = [col for col in df.columns if col.endswith("_pass_filter")]
    if not filter_cols:
        print("No filter columns found for summary plot."); return
    filter_names = [col.replace("_pass_filter", "") for col in filter_cols]
    true_counts = [int(df[col].sum()) for col in filter_cols]
    false_counts = [int(len(df[col]) - df[col].sum()) for col in filter_cols]
    x = range(len(filter_names)); width = 0.4
    plt.figure(figsize=(14,8))
    plt.bar([i - width/2 for i in x], true_counts, width=width, label="Pass")
    plt.bar([i + width/2 for i in x], false_counts, width=width, label="Fail")
    plt.xticks(ticks=list(x), labels=filter_names, rotation=45, ha='right'); plt.ylabel("Number of Molecules"); plt.title("Structural Alert Filter Summary"); plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.savefig(output_file, dpi=300, bbox_inches='tight'); plt.close()
    vprint(f"Saved {output_file}")

def _mol_to_pdb_block(mol, mol_name: str = "unknown"):
    """Convert molecule to PDB block with robust error handling."""
    if mol is None:
        return ""
    if mol.GetNumConformers() == 0:
        # Try to generate a conformer
        try:
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            AllChem.UFFOptimizeMolecule(mol)
        except Exception as e:
            print(f"Failed to embed {mol_name}: {e}")
            return ""    
 
    try:
            
        # Get conformer and validate 3D coordinates
        conf = mol.GetConformer()
        
        # Check if coordinates are meaningful
        coords_valid = False
        for i in range(mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            if abs(pos.x) > 0.001 or abs(pos.y) > 0.001 or abs(pos.z) > 0.001:
                coords_valid = True
                break
                
        if not coords_valid:
            print(f"Error: {mol_name} - All coordinates are zero")
            return ""
        
        # Ensure conformer is marked as 3D
        conf.Set3D(True)
        
        # Create PDB block without adding hydrogens preserving the original structure)
        try:
            pdb_block = Chem.MolToPDBBlock(mol, confId=0)
        except Exception as e:
            print(f"Error: {mol_name} - Failed to generate PDB: {e}")
            # Try with hydrogen addition as fallback
            try:
                mol_h = Chem.AddHs(mol, addCoords=True)
                pdb_block = Chem.MolToPDBBlock(mol_h, confId=0)
                vprint(f"Success: {mol_name} - Generated PDB with added hydrogens")
            except Exception as e2:
                print(f"Error: {mol_name} - Fallback also failed: {e2}")
                return ""
        
        # Validate PDB block
        if not pdb_block or len(pdb_block.strip()) < 50:
            print(f"Error: {mol_name} - Generated PDB block too short ({len(pdb_block)} chars)")
            return ""
            
        # Count actual coordinate lines
        coord_lines = [line for line in pdb_block.split('\n') 
                      if line.startswith(('ATOM', 'HETATM')) and len(line) >= 54]
        
        if len(coord_lines) < mol.GetNumAtoms() * 0.5:  # At least 50% of atoms should have coordinates
            print(f"Error: {mol_name} - Too few coordinate lines ({len(coord_lines)} vs {mol.GetNumAtoms()} atoms)")
            return ""
            
        vprint(f"Success: {mol_name} - Generated PDB with {len(coord_lines)} coordinate lines")
        return pdb_block
        
    except Exception as e:
        print(f"Error: {mol_name} - Exception in PDB generation: {e}")
        return ""

def debug_molecule_coords(mol, name: str):
    """Comprehensive molecule coordinate debugging."""
    vprint(f"\n=== Debugging {name} ===")
    vprint(f"Conformers: {mol.GetNumConformers()}")
    vprint(f"Atoms: {mol.GetNumAtoms()}")
    
    if mol.GetNumConformers() > 0:
        conf = mol.GetConformer()
        vprint(f"Conformer is 3D: {conf.Is3D()}")
        
        # Check first few atom coordinates
        for i in range(min(3, mol.GetNumAtoms())):
            pos = conf.GetAtomPosition(i)
            atom = mol.GetAtomWithIdx(i)
            vprint(f"  Atom {i} ({atom.GetSymbol()}): ({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f})")
        
        # Check coordinate ranges
        x_coords = [conf.GetAtomPosition(i).x for i in range(mol.GetNumAtoms())]
        y_coords = [conf.GetAtomPosition(i).y for i in range(mol.GetNumAtoms())]
        z_coords = [conf.GetAtomPosition(i).z for i in range(mol.GetNumAtoms())]
        
        vprint(f"  X range: {min(x_coords):.3f} to {max(x_coords):.3f}")
        vprint(f"  Y range: {min(y_coords):.3f} to {max(y_coords):.3f}")
        vprint(f"  Z range: {min(z_coords):.3f} to {max(z_coords):.3f}")
        
        # Check if all coordinates are zero
        all_zero = all(abs(x) < 0.001 and abs(y) < 0.001 and abs(z) < 0.001 
                      for x, y, z in zip(x_coords, y_coords, z_coords))
        vprint(f"  All coordinates near zero: {all_zero}")
    
    # Test PDB generation
    pdb = _mol_to_pdb_block(mol, name)
    vprint(f"PDB generation successful: {len(pdb) > 50}")
    
    return len(pdb) > 50



def create_top_compounds_dashboard(df: pd.DataFrame, output_file: str, protein_pdb: Optional[str]=None):
    top_compounds = df.nlargest(20, 'Composite_Score').copy()
    top5 = top_compounds.head(10).copy().reset_index(drop=True)
    
    pdb_blocks = []
    for i, mol in enumerate(top5['mol']):
        vprint(f"Processing molecule {i+1}: {top5.iloc[i]['name']}")
        vprint(f"  - Has {mol.GetNumConformers()} conformer(s)")
        vprint(f"  - Has {mol.GetNumAtoms()} atoms")
    
        pdb_block = _mol_to_pdb_block(mol, mol.GetProp("_Name") if mol.HasProp("_Name") else f"mol_{i}")
        if pdb_block:
            vprint(f"  - Generated PDB block: {len(pdb_block)} characters")
        else:
            print(f"  - Failed to generate PDB block")
            pdb_block = ""  # Ensure we have a string, not None
    
        pdb_blocks.append(pdb_block)
    
    top5['pdb_block'] = pdb_blocks

    fig = make_subplots(rows=3, cols=2, subplot_titles=('Top 20 Compounds Ranking','Property Comparison Radar','Binding Affinity vs Drug-likeness','Lipinski Violations Distribution','Filter Warnings Table','Filter Warnings'), specs=[[{"colspan":2},None],[{"type":"scatterpolar"},{"type":"scatter"}],[{"type":"domain"},{"type":"heatmap"}]], vertical_spacing=0.08, horizontal_spacing=0.1)
    top10 = top_compounds.head(10)
    colors = ['gold' if i==0 else 'silver' if i==1 else 'chocolate' if i==2 else 'lightblue' for i in range(len(top10))]
    fig.add_trace(go.Bar(x=top10['name'], y=top10['Composite_Score'], marker_color=colors, text=[f"#{int(r)}" for r in top10['Rank']], textposition='auto', name='Composite Score', hovertemplate='<b>%{x}</b><br>Composite Score: %{y:.3f}<br>Rank: %{text}<extra></extra>'), row=1, col=1)
    properties = ['Affinity_Score','QED_Score','SA_Score_Norm','Lipinski_Score','Filter_Score']
    prop_labels = ['Binding Affinity','Drug-likeness','Synthetic Access.','Lipinski','Filter Pass']
    for _, compound in top_compounds.head(10).iterrows():
        fig.add_trace(go.Scatterpolar(r=[compound.get(p,0) for p in properties], theta=prop_labels, fill='toself', name=f"#{int(compound['Rank'])}: {compound['name']}", opacity=0.6), row=2, col=1)
    fig.add_trace(go.Scatter(x=top_compounds['Affinity_Numeric'], y=top_compounds['QED'], mode='markers+text', marker=dict(size=(top_compounds['Composite_Score']*20).clip(8,30), color=top_compounds['Rank'], colorscale='Viridis', showscale=False, line=dict(width=1,color='white')), text=[f"#{int(r)}" for r in top_compounds['Rank']], textposition='middle center', name='Compounds', customdata=np.stack([top_compounds['name'], top_compounds['Composite_Score'], top_compounds.index], axis=1), hovertemplate='<b>%{customdata[0]}</b><br>Affinity: %{x:.2f}<br>QED: %{y:.3f}<br>Score: %{customdata[1]:.3f}<extra></extra>'), row=2, col=2)

    # Calculate distribution of Lipinski violations across all molecules
    lipinski_distribution = df['Lipinski_Violations'].value_counts().sort_index()
    violation_labels = [f"{int(v)} violations" if v != 1 else "1 violation" for v in lipinski_distribution.index]
    fig.add_trace(go.Pie(values=lipinski_distribution.values, labels=violation_labels, textinfo='label+percent', hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>', showlegend=True), row=3, col=1)

    filter_columns = [c for c in df.columns if c.endswith('_pass_filter')]
    if filter_columns:
        filter_data = top_compounds.head(10)[filter_columns].astype(int).values
        filter_names = [c.replace('_pass_filter','') for c in filter_columns]
        compound_names = [n for n in top_compounds.head(10)['name']]
    
        # Create a mapping of compound names to their reasons
        compound_reasons_map = {}
        for idx, row in top_compounds.head(10).iterrows():
            full_name = row['name']
            short_name = full_name
            compound_reasons_map[short_name] = {}
            compound_reasons_map[full_name] = {}  # Also store with full name
        
            for filter_name in filter_names:
                reasons_col = f"{filter_name}_reasons"
                if reasons_col in row.index:
                    reason = str(row[reasons_col]) if not pd.isna(row[reasons_col]) else "No reason available"
                    compound_reasons_map[short_name][filter_name] = reason.replace(";", ", ")
                    compound_reasons_map[full_name][filter_name] = reason.replace(";", ", ")
    
        # Prepare reasons data for hovertemplate
        reasons_data = []
        for filter_name in filter_names:
            reasons_col = f"{filter_name}_reasons"
            if reasons_col in top_compounds.columns:
                compound_reasons = []
                for idx, row in top_compounds.head(10).iterrows():
                    reason = row.get(reasons_col, "No reason available")
                    # Truncate long reasons for preview
                    if isinstance(reason, str) and len(reason) > 100:
                        preview = reason[:100] + "... (click to see full)"
                        compound_reasons.append(f"Preview: {preview}")
                    else:
                        compound_reasons.append(str(reason))
                reasons_data.append(compound_reasons)
            else:
                reasons_data.append(["No reason data"] * 10)
    
        # Create hover text
        hover_text = []
        for i, filter_name in enumerate(filter_names):
            filter_hover = []
            for j, compound_name in enumerate(compound_names):
                pass_status = "Pass" if filter_data.T[i][j] == 1 else "Fail"
                base_text = f"Compound: {compound_name}<br>Filter: {filter_name}<br>Status: {pass_status}"
            
                # Add reason if filter failed
                if filter_data.T[i][j] == 0:  # Failed filter
                    reason = reasons_data[i][j] if i < len(reasons_data) else "No reason available"
                    base_text += f"<br>Reason: {reason}"
            
                filter_hover.append(base_text)
            hover_text.append(filter_hover)
    
        fig.add_trace(go.Heatmap(
            z=filter_data.T,
            x=compound_names,
            y=filter_names,
            colorscale=[[0,'red'],[1,'green']],
            showscale=False,
            hovertext=hover_text,
            hovertemplate='%{hovertext}<extra></extra>'
        ), row=3, col=2)


    fig.update_layout(
        height=1400,
        title_text=f"<b>Top Molecular Candidates Dashboard</b><br><sup>Top compounds interactive dashboard</sup>",
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.5,
            xanchor='left',
            yanchor='middle',
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='rgba(0, 0, 0, 0.2)',
            borderwidth=1,
            font=dict(size=10)
        )
    )
    fig_file = output_file.replace('.html','__plots.html')
    fig.write_html(fig_file, include_plotlyjs='cdn', full_html=True, config={'displayModeBar': True})
    with open(fig_file, 'r', encoding='utf-8') as f:
        html = f.read()
    
    top5_json = []
    top5_reasons = {}
    for i, row in top5.iterrows():
        top5_json.append({'idx': int(row.name),'rank': int(row['Rank']), 'name': str(row['name']), 'score': float(row['Composite_Score']), 'affinity': float(row['Affinity_Numeric']) if not pd.isna(row['Affinity_Numeric']) else None, 'mw': float(row['MW']) if not pd.isna(row['MW']) else None, 'logp': float(row['LogP']) if not pd.isna(row['LogP']) else None, 'qed': float(row['QED']) if not pd.isna(row['QED']) else None, 'lipinski_violations': int(row['Lipinski_Violations']) if not pd.isna(row['Lipinski_Violations']) else None, 'filter_score': float(row['Filter_Score']) if not pd.isna(row['Filter_Score']) else None, 'smiles': str(row['smiles']), 'pdb_block': row['pdb_block']})
    compound_reasons_json_str = json.dumps(compound_reasons_map).replace('</','<\/')
    top5_json_str = json.dumps(top5_json).replace('</','<\/')

    for filter_name in filter_names:
        reasons_col = f"{filter_name}_reasons"
        if reasons_col in top5.columns:
            top5_reasons[filter_name] = top5[reasons_col].fillna("No reason available").tolist()

    
    protein_pdb_text = ""
    if protein_pdb and os.path.isfile(protein_pdb):
        try:
            protein_pdb_text = Path(protein_pdb).read_text()
            protein_pdb_text = protein_pdb_text.replace('</','<\/')
        except Exception as e:
            print(f"Failed to read protein PDB: {e}")
            protein_pdb_text = ""
    
    reasons_json_str = json.dumps(top5_reasons).replace('</','<\/')

    inject_block = f"""
<!-- BEGIN: custom 3Dmol.js viewers and cross-plot interactivity -->
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"/>
<div style="padding:10px; background:#f7f7f9; border-radius:6px; margin:10px;">
  <h2 style="margin:6px 0;">Top 10 3D Viewers</h2>
  <div style="margin-bottom:8px;">
    <button id="dock-toggle" style="padding:6px 10px; border-radius:6px; cursor:pointer;">Dock (show protein)</button>
    <span style="margin-left:12px; color:#666;">Click any plot item to highlight across plots and viewers.</span>
  </div>
  <div id="viewers" style="display:flex; gap:8px; flex-wrap:wrap;">
    {"".join([f'<div id="viewer{i}" style="width:300px; height:300px; border:1px solid #ddd; border-radius:6px; position:relative;"></div>' for i in range(10)])}
  </div>
  <div id="top5-table" style="margin-top:12px;"></div>
</div>
<script src="https://3Dmol.csb.pitt.edu/build/3Dmol-min.js"></script>
<script>
const top5 = {top5_json_str};
const protein_pdb = {json.dumps(protein_pdb_text)};
const compoundReasonsMap = {compound_reasons_json_str};
const top5Reasons = {reasons_json_str};

let viewers = [];
let protein_added = false;
let selectedIndex = null;

// store original Plotly trace marker sizes/colors (per graphDiv)
const plotlyOriginals = new WeakMap();

// Save the original marker sizes/colors for a given graphDiv
function saveOriginalPlotlyStyles(graphDiv) {{
    if (!graphDiv || plotlyOriginals.has(graphDiv)) return;
    const orig = [];
    if (!graphDiv.data) {{ plotlyOriginals.set(graphDiv, orig); return; }}
    graphDiv.data.forEach(trace => {{
        const n = (trace.x && trace.x.length) ||
                  (trace.y && trace.y.length) ||
                  (trace.customdata && trace.customdata.length) || 0;
        // Normalize sizes
        let sizes;
        if (trace.marker && typeof trace.marker.size !== 'undefined') {{
            sizes = Array.isArray(trace.marker.size) ? trace.marker.size.slice() : Array(n).fill(trace.marker.size);
        }} else {{
            sizes = Array(n).fill(8);
        }}
        // Normalize colors
        let colors;
        if (trace.marker && typeof trace.marker.color !== 'undefined') {{
            colors = Array.isArray(trace.marker.color) ? trace.marker.color.slice() : Array(n).fill(trace.marker.color);
        }} else {{
            colors = Array(n).fill('lightblue');
        }}
        orig.push({{ sizes, colors, type: trace.type || '' }});
    }});
    plotlyOriginals.set(graphDiv, orig);
}}

// Deselect everything -- restore original visuals
function deselectAll() {{
    selectedIndex = null;

    // reset table rows
    for (let i = 0; i < top5.length; i++) {{
        const row = document.getElementById(`top5row${{i}}`);
        if (row) row.style.backgroundColor = 'white';
    }}

    // reset 3D viewers to default style
    for (let i = 0; i < viewers.length; i++) {{
        if (!viewers[i]) continue;
        try {{
            if (protein_added) {{
                // if protein is shown, re-apply the "protein + ligand" look used elsewhere
                viewers[i].setStyle({{ model: 0 }}, {{ stick: {{ radius: 0.3, colorscheme: 'greenCarbon', opacity: 1 }}}});
                viewers[i].setStyle({{ model: -1 }}, {{ cartoon: {{ color: 'spectrum', opacity: 0.8 }}}});
            }} else {{
                // ligand-only default
                viewers[i].setStyle({{}}, {{ stick: {{ radius: 0.2, colorscheme: 'default', opacity: 1 }}}});
            }}
            viewers[i].render();
        }} catch (e) {{
            console.error(`reset viewer ${{i}} error:`, e);
        }}
    }}

    // restore Plotly traces to their original marker sizes/colors
    highlightInPlotlyGraphs(null, null);
}}

// Toggle selection for an index
function toggleSelectCompound(index) {{
    if (selectedIndex === index) {{
        deselectAll();
    }} else {{
        selectCompound(index);
    }}
}}

// Keep existing selectCompound behaviour but rely on toggleSelectCompound to toggle
function selectCompound(index) {{
    if (index < 0 || index >= top5.length) return;

    selectedIndex = index;
    const selectedMolecule = top5[index];

    console.log(`Selecting compound ${{index}}: ${{selectedMolecule.name}}`);

    // Highlight table row
    for (let i = 0; i < top5.length; i++) {{
        const row = document.getElementById(`top5row${{i}}`);
        if (row) {{
            row.style.backgroundColor = (i === index) ? '#fff3cd' : 'white';
        }}
    }}

    // Highlight 3D viewers: selected viewer emphasized, others dimmed
    for (let i = 0; i < viewers.length; i++) {{
        if (!viewers[i]) continue;
        try {{
            if (i === index) {{
                // selected
                if (protein_added) {{
                    viewers[i].setStyle({{ model: 0 }}, {{ stick: {{ radius: 0.4, colorscheme: 'orangeCarbon' }}}});
                }} else {{
                    viewers[i].setStyle({{}}, {{ stick: {{ radius: 0.3, colorscheme: 'orangeCarbon' }}}});
                }}
                viewers[i].zoomTo();
            }} else {{
                // dim others
                if (protein_added) {{
                    viewers[i].setStyle({{ model: 0 }}, {{ stick: {{ radius: 0.2, opacity: 0.5 }}}});
                }} else {{
                    viewers[i].setStyle({{}}, {{ stick: {{ radius: 0.15, opacity: 0.5 }}}});
                }}
            }}
            viewers[i].render();
        }} catch (e) {{
            console.error(`Error highlighting viewer ${{i}}:`, e);
        }}
    }}

    // Highlight plotly graphs (uses saved originals to compute baseline)
    highlightInPlotlyGraphs(selectedMolecule.name, index);
}}

// Highlight selected compound in all Plotly graphs, or restore originals if moleculeName is null
function highlightInPlotlyGraphs(moleculeName, selectedIdx) {{
    const graphs = document.querySelectorAll('.plotly-graph-div');

    graphs.forEach(graphDiv => {{
        try {{
            if (!graphDiv.data) return;

            // Ensure we have stored originals
            saveOriginalPlotlyStyles(graphDiv);
            const orig = plotlyOriginals.get(graphDiv) || [];

            for (let traceIndex = 0; traceIndex < graphDiv.data.length; traceIndex++) {{
                const trace = graphDiv.data[traceIndex];
                // If no moleculeName -> restore original marker sizes/colors for this trace
                if (!moleculeName) {{
                    const o = orig[traceIndex];
                    if (o) {{
                        Plotly.restyle(graphDiv, {{
                            'marker.size': [o.sizes],
                            'marker.color': [o.colors]
                        }}, [traceIndex]);
                    }}
                    continue;
                }}

                // handle marker scatter traces with customdata
                if (trace.mode && trace.mode.includes('markers') && trace.customdata) {{
                    const baseSizes = (orig[traceIndex] && orig[traceIndex].sizes) ? orig[traceIndex].sizes : Array(trace.customdata.length).fill(8);
                    const baseColors = (orig[traceIndex] && orig[traceIndex].colors) ? orig[traceIndex].colors : Array(trace.customdata.length).fill('lightblue');
                    const sizeUpdate = [];
                    const colorUpdate = [];

                    for (let pointIndex = 0; pointIndex < trace.customdata.length; pointIndex++) {{
                        const pointName = Array.isArray(trace.customdata[pointIndex]) ? trace.customdata[pointIndex][0] : trace.customdata[pointIndex];
                        if (pointName === moleculeName) {{
                            // emphasize selected point (bigger, red)
                            sizeUpdate.push(Math.max(baseSizes[pointIndex] || 8, 35));
                            colorUpdate.push('red');
                        }} else {{
                            // restore to baseline
                            sizeUpdate.push(baseSizes[pointIndex] || 8);
                            colorUpdate.push(baseColors[pointIndex] || 'lightblue');
                        }}
                    }}

                    Plotly.restyle(graphDiv, {{
                        'marker.size': [sizeUpdate],
                        'marker.color': [colorUpdate]
                    }}, [traceIndex]);
                }}
                // handle bar traces (color whole bar)
                else if (trace.type === 'bar' && trace.x) {{
                    const baseColors = (orig[traceIndex] && orig[traceIndex].colors) ? orig[traceIndex].colors : Array(trace.x.length).fill('lightblue');
                    const colors = [];
                    for (let pointIndex = 0; pointIndex < trace.x.length; pointIndex++) {{
                        colors.push(trace.x[pointIndex] === moleculeName ? 'red' : baseColors[pointIndex] || 'lightblue');
                    }}
                    Plotly.restyle(graphDiv, {{ 'marker.color': [colors] }}, [traceIndex]);
                }}
                // other trace types: skip / leave as-is
            }}
        }} catch (error) {{
            console.log('Error highlighting in Plotly graph:', error);
        }}
    }});
}}

// Attach click handlers to Plotly graphs and save original styles on load
function attachPlotlyClickHandlers() {{
    // Wait for graphs to be rendered
    setTimeout(() => {{
        const graphs = document.querySelectorAll('.plotly-graph-div');
        console.log(`Found ${{graphs.length}} Plotly graphs`);

        graphs.forEach(graphDiv => {{
            try {{
                // save original style once for this graph
                saveOriginalPlotlyStyles(graphDiv);
            }} catch(e) {{
                console.warn('Failed to save original styles for a graph:', e);
            }}

            // attach click handler
            graphDiv.on('plotly_click', function(eventData) {{
                try {{
                    const point = eventData.points && eventData.points[0];
                    if (!point) return;
                    let moleculeName = null;

                    // Extract molecule name from point.customdata or point.x
                    if (point.customdata) {{
                        moleculeName = Array.isArray(point.customdata) ? point.customdata[0] : point.customdata;
                    }} else if (typeof point.x !== 'undefined') {{
                        moleculeName = point.x;
                    }}

                    if (!moleculeName) return;

                    // Try match to top5 by name
                    for (let i = 0; i < top5.length; i++) {{
                        if (top5[i].name === moleculeName) {{
                            // toggle selection: if already selected, deselect; otherwise select
                            if (selectedIndex === i) deselectAll();
                            else selectCompound(i);
                            return;
                        }}
                    }}

                    // fallback: match by rank embedded in text
                    if (point.text) {{
                        for (let i = 0; i < top5.length; i++) {{
                            if (point.text.includes(`#${{top5[i].rank}}`)) {{
                                if (selectedIndex === i) deselectAll();
                                else selectCompound(i);
                                return;
                            }}
                        }}
                    }}
                }} catch (error) {{
                    console.log('Error in Plotly click handler:', error);
                }}
            }});
        }});
    }}, 1000);
}}

// Function to show full reason in a modal/popup
function showFullReason(filterName, compoundName, fullReason) {{
    // Remove any existing modals first
    const existingModal = document.querySelector('.reason-modal');
    if (existingModal) existingModal.remove();
    
    // Create modal overlay
    const modal = document.createElement('div');
    modal.className = 'reason-modal';
    modal.style.cssText = `
        position: fixed; top: 0; left: 0; width: 100%; height: 100%; 
        background: rgba(0,0,0,0.7); z-index: 10000; 
        display: flex; align-items: center; justify-content: center;
    `;
    
    const modalContent = document.createElement('div');
    modalContent.style.cssText = `
        background: white; padding: 20px; border-radius: 8px; 
        max-width: 80%; max-height: 80%; overflow-y: auto;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        position: relative;
    `;
    
    modalContent.innerHTML = `
        <h3 style="margin-top: 0; color: #333;">Filter Failure Reason</h3>
        <p><strong>Compound:</strong> ${{compoundName}}</p>
        <p><strong>Filter:</strong> ${{filterName}}</p>
        <p><strong>Reason:</strong> ${{fullReason}}</p>
        </div>
        <button id="close-reason-modal" 
                style="padding: 8px 16px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer;">
            Close
        </button>
    `;
    
    modal.appendChild(modalContent);
    document.body.appendChild(modal);
    
    // Add event listeners for closing
    const closeBtn = modal.querySelector('#close-reason-modal');
    closeBtn.addEventListener('click', function(e) {{
        e.preventDefault();
        e.stopPropagation();
        modal.remove();
    }});
    
    // Close on overlay click (but not on content click)
    modal.addEventListener('click', function(e) {{
        if (e.target === modal) {{
            modal.remove();
        }}
    }});
    
    // Close on Escape key
    const escapeHandler = function(e) {{
        if (e.key === 'Escape') {{
            modal.remove();
            document.removeEventListener('keydown', escapeHandler);
        }}
    }};
    document.addEventListener('keydown', escapeHandler);
}}

// Enhanced click handler for heatmap
function attachHeatmapClickHandler() {{
    setTimeout(() => {{
        const graphs = document.querySelectorAll('.plotly-graph-div');
        graphs.forEach(graphDiv => {{
            graphDiv.on('plotly_click', function(eventData) {{
                try {{
                    const point = eventData.points && eventData.points[0];
                    if (!point || !point.data || point.data.type !== 'heatmap') return;
                    
                    // Check if this is a failed filter (red square)
                    if (point.z === 0) {{  // Failed filter
                        const compoundName = point.x;
                        const filterName = point.y;
                        
                        console.log(`Clicked on failed filter: ${{filterName}} for compound: ${{compoundName}}`);
                        
                        // Get the reason from the compound reasons map
                        let fullReason = "No reason available";
                        
                        if (compoundReasonsMap[compoundName] && compoundReasonsMap[compoundName][filterName]) {{
                            fullReason = compoundReasonsMap[compoundName][filterName];
                        }} else {{
                            // Try to find by partial name match
                            for (const [storedName, reasonsData] of Object.entries(compoundReasonsMap)) {{
                                if (storedName.includes(compoundName) || compoundName.includes(storedName.substring(0, 10))) {{
                                    if (reasonsData[filterName]) {{
                                        fullReason = reasonsData[filterName];
                                        break;
                                    }}
                                }}
                            }}
                        }}
                        
                        console.log(`Found reason: ${{fullReason.substring(0, 100)}}...`);
                        showFullReason(filterName, compoundName, fullReason);
                    }}
                }} catch (error) {{
                    console.error('Error in heatmap click handler:', error);
                }}
            }});
        }});
    }}, 1500);
}}
// Show error message in viewer container
function showMoleculeError(element, index, message) {{
    element.innerHTML = `
        <div style="
            display: flex; 
            align-items: center; 
            justify-content: center; 
            height: 100%; 
            background: #f8f9fa; 
            color: #666; 
            text-align: center;
            font-size: 12px;
            padding: 10px;
        ">
            <div>
                <i class="fas fa-exclamation-triangle" style="font-size: 24px; margin-bottom: 8px; color: #ffc107;"></i>
                <div style="font-weight: bold;">Molecule #${{index + 1}}</div>
                <div style="margin-top: 4px;">${{message}}</div>
            </div>
        </div>
    `;
}}

// Add molecule information overlay
function addMoleculeInfo(element, molecule, index) {{
    const info = document.createElement('div');
    info.style.cssText = `
        position: absolute; 
        top: 5px; 
        left: 5px; 
        background: rgba(0,0,0,0.7); 
        color: white; 
        padding: 5px 8px; 
        border-radius: 3px; 
        font-size: 11px; 
        font-family: monospace;
        z-index: 1000;
        max-width: 200px;
        pointer-events: none;
    `;
    info.innerHTML = `
        <div style="font-weight: bold;">#${{molecule.rank}}: ${{molecule.name}}</div>
        <div>Score: ${{molecule.score.toFixed(3)}}</div>
        ${{molecule.affinity !== null ? `<div>Affinity: ${{molecule.affinity.toFixed(2)}}</div>` : ''}}
    `;
    element.appendChild(info);
}}

// Initialize 3D molecular viewers
function initViewers() {{
    console.log('Initializing viewers for', top5.length, 'molecules');

    // Clear viewers array
    viewers = [];

    for (let i = 0; i < 10; i++) {{
        const element = document.getElementById(`viewer${{i}}`);
        if (!element) {{
            console.error(`Viewer element ${{i}} not found in DOM`);
            viewers.push(null);
            continue;
        }}
        
        try {{
            // Clear any existing content
            element.innerHTML = '';
            
            // Check if we have molecule data
            if (i >= top5.length || !top5[i]) {{
                showMoleculeError(element, i, 'No molecule data available');
                viewers.push(null);
                continue;
            }}
            
            const molecule = top5[i];
            
            // Validate PDB data
            if (!molecule.pdb_block || molecule.pdb_block.length < 50) {{
                console.warn(`Molecule ${{i}} (${{molecule.name}}) has insufficient PDB data`);
                showMoleculeError(element, i, 'No 3D coordinate data available');
                viewers.push(null);
                continue;
            }}
            
            // Create viewer
            const viewer = $3Dmol.createViewer(element, {{
                backgroundColor: 'white',
                antialias: true,
                quality: 'medium'
            }});
            
            if (!viewer) {{
                throw new Error(`Failed to create 3Dmol viewer`);
            }}
            
            console.log(`Loading molecule ${{i}}: ${{molecule.name}}`);
            
            try {{
                // Add model with validation
                const model = viewer.addModel(molecule.pdb_block, 'pdb');
                if (!model) {{
                    throw new Error('Failed to parse PDB data');
                }}
                
                // Set styling for ligand
                viewer.setStyle({{}}, {{
                    stick: {{
                        radius: 0.2,
                        colorscheme: 'default'
                    }}
                }});
                
                // Zoom and render
                viewer.zoomTo();
                viewer.render();
                
                // Add to viewers array
                viewers.push(viewer);
                
                console.log(`Successfully loaded molecule ${{i}}: ${{molecule.name}}`);
                
                // Add molecule info overlay
                addMoleculeInfo(element, molecule, i);
                
                // Add click handler with drag detection to avoid conflict with rotation

                element.style.cursor = 'pointer';

                let pointerDown = false;
                let startX = 0, startY = 0;
                let isDrag = false;
                const DRAG_THRESHOLD = 6; // pixels; increase to tolerate more movement before counting as drag

                element.addEventListener('pointerdown', function(ev) {{
                    // Start tracking; do NOT stop propagation so 3Dmol keeps handling rotation
                    pointerDown = true;
                    isDrag = false;
                    // Use clientX/Y so it works for mouse & touch
                    startX = ev.clientX;
                    startY = ev.clientY;
                }}, {{passive: true}});

                element.addEventListener('pointermove', function(ev) {{
                    if (!pointerDown) return;
                    const dx = ev.clientX - startX;
                    const dy = ev.clientY - startY;
                    if (!isDrag && (dx*dx + dy*dy) > (DRAG_THRESHOLD * DRAG_THRESHOLD)) {{
                        isDrag = true;
                    }}
                }}, {{passive: true}});

                element.addEventListener('pointerup', function(ev) {{
                    // If it was a click (no significant movement), toggle selection
                    if (!isDrag) {{
                        try {{
                            toggleSelectCompound(i);
                        }} catch (e) {{
                            console.error('Selection handler failed:', e);
                        }}
                    }}
                    // reset tracking
                    pointerDown = false;
                    isDrag = false;
                }}, {{passive: true}});

                element.addEventListener('pointercancel', function() {{
                    pointerDown = false;
                    isDrag = false;
                }}, {{passive: true}});

                
            }} catch (modelError) {{
                console.error(`Failed to load molecule ${{i}}:`, modelError);
                showMoleculeError(element, i, `Failed to load: ${{modelError.message}}`);
                viewers.push(null);
            }}
            
        }} catch (error) {{
            console.error(`Failed to initialize viewer ${{i}}:`, error);
            showMoleculeError(element, i, `Viewer error: ${{error.message}}`);
            viewers.push(null);
        }}
    }}
    
    console.log(`Initialized ${{viewers.filter(v => v).length}} out of ${{viewers.length}} viewers`);
}}

// Toggle protein display in all viewers


// Build HTML table of top 10 compounds
function buildTop5Table() {{
    const container = document.getElementById('top5-table');
    let html = '<table style="width:100%; border-collapse:collapse; margin-top: 10px;">';
    html += '<thead><tr style="background:#e9ecef;"><th style="padding:8px;border:1px solid #ddd;">Rank</th><th style="padding:8px;border:1px solid #ddd;">Name</th><th style="padding:8px;border:1px solid #ddd;">Score</th><th style="padding:8px;border:1px solid #ddd;">Affinity</th><th style="padding:8px;border:1px solid #ddd;">MW</th><th style="padding:8px;border:1px solid #ddd;">LogP</th><th style="padding:8px;border:1px solid #ddd;">QED</th><th style="padding:8px;border:1px solid #ddd;">Lipinski Violations</th><th style="padding:8px;border:1px solid #ddd;">Filter Score</th></tr></thead><tbody>';
    
    for (let i = 0; i < top5.length; i++) {{
        const molecule = top5[i];
        html += `
            <tr id="top5row${{i}}" onclick="toggleSelectCompound(${{i}})" style="cursor:pointer; transition: background-color 0.2s;">
                <td style="padding:8px;border:1px solid #ddd; text-align:center; font-weight:bold;">${{molecule.rank}}</td>
                <td style="padding:8px;border:1px solid #ddd;">${{molecule.name}}</td>
                <td style="padding:8px;border:1px solid #ddd; text-align:center;">${{molecule.score.toFixed(3)}}</td>
                <td style="padding:8px;border:1px solid #ddd; text-align:center;">${{molecule.affinity !== null ? molecule.affinity.toFixed(2) : 'N/A'}}</td>
                <td style="padding:8px;border:1px solid #ddd; text-align:center;">${{molecule.mw !== null ? molecule.mw.toFixed(1) : 'N/A'}}</td>
                <td style="padding:8px;border:1px solid #ddd; text-align:center;">${{molecule.logp !== null ? molecule.logp.toFixed(2) : 'N/A'}}</td>
                <td style="padding:8px;border:1px solid #ddd; text-align:center;">${{molecule.qed !== null && molecule.qed !== undefined ? molecule.qed.toFixed(3) : 'N/A'}}</td>
                <td style="padding:8px;border:1px solid #ddd; text-align:center;">${{molecule.lipinski_violations !== null && molecule.lipinski_violations !== undefined ? molecule.lipinski_violations : 'N/A'}}</td>
                <td style="padding:8px;border:1px solid #ddd; text-align:center;">${{molecule.filter_score !== null && molecule.filter_score !== undefined ? molecule.filter_score.toFixed(3) : 'N/A'}}</td>
            </tr>
        `;
    }}
    html += '</tbody></table>';
    container.innerHTML = html;
}}

function toggleProtein() {{
    if (!protein_pdb || protein_pdb.length < 10) {{
        alert('No protein PDB provided. Use --protein parameter to include protein structure.');
        return;
    }}

    console.log('Toggling protein, current state:', protein_added);

    if (!protein_added) {{
        // Add protein + surface to all viewers
        for (let i = 0; i < viewers.length; i++) {{
            if (viewers[i] && top5[i] && top5[i].pdb_block) {{
                try {{
                    console.log(`Adding protein to viewer ${{i}}`);

                    // Add protein model (appended as the last model)
                    viewers[i].addModel(protein_pdb, 'pdb');

                    // Add a VDW surface for the last-added model (use model: -1)
                    viewers[i].addSurface(
                        $3Dmol.SurfaceType.VDW,
                        {{ opacity: 0.85, color: 'white' }},
                        {{ model: -1 }} // -1 selects the most recently added model
                    );

                    // Optional: show protein as cartoon too (linked to the last model)
                    viewers[i].setStyle({{ model: -1 }}, {{ cartoon: {{ color: 'spectrum', opacity: 0.8 }} }});

                    // Keep ligand as sticks (model 0)
                    viewers[i].setStyle({{ model: 0 }}, {{ stick: {{ radius: 0.3, colorscheme: 'greenCarbon' }} }});

                    viewers[i].zoomTo();
                    viewers[i].render();
                }} catch (e) {{
                    console.error(`Failed to add protein to viewer ${{i}}:`, e);
                }}
            }}
        }}
        protein_added = true;
        document.getElementById('dock-toggle').textContent = 'Hide protein';
    }} else {{
        // Remove protein/surface, keep only ligand
        for (let i = 0; i < viewers.length; i++) {{
            if (viewers[i] && top5[i] && top5[i].pdb_block) {{
                try {{
                    console.log(`Removing protein from viewer ${{i}}`);

                    // Remove any surfaces first (important!)
                    if (typeof viewers[i].removeAllSurfaces === 'function') {{
                        viewers[i].removeAllSurfaces();
                    }}

                    // Then remove models
                    if (typeof viewers[i].removeAllModels === 'function') {{
                        viewers[i].removeAllModels();
                    }} else {{
                        // Fall back: try clearing models another way if API differs
                        viewers[i].clear();
                    }}

                    // Re-add ligand-only model (model 0)
                    viewers[i].addModel(top5[i].pdb_block, 'pdb');
                    viewers[i].setStyle({{}}, {{ stick: {{ radius: 0.2, colorscheme: 'default' }} }});
                    viewers[i].zoomTo();
                    viewers[i].render();
                }} catch (e) {{
                    console.error(`Failed to restore ligand in viewer ${{i}}:`, e);
                }}
            }}
        }}
        protein_added = false;
        document.getElementById('dock-toggle').textContent = 'Dock (show protein)';
    }}
}}


// Wait for 3Dmol.js to load and initialize everything
function waitFor3Dmol() {{
    if (typeof $3Dmol !== 'undefined' && $3Dmol.createViewer) {{
        console.log('3Dmol.js loaded successfully, initializing viewers...');
        initViewers();
        buildTop5Table();
        
        // Set up dock button
        const dockButton = document.getElementById('dock-toggle');
        if (dockButton) {{
            dockButton.addEventListener('click', toggleProtein);
        }}
        
        // Attach Plotly handlers after a delay
        attachPlotlyClickHandlers();
        attachHeatmapClickHandler();
        
    }} else {{
        console.log('Waiting for 3Dmol.js to load...');
        setTimeout(waitFor3Dmol, 200);
    }}
}}

// Initialize everything when DOM is ready
document.addEventListener('DOMContentLoaded', function() {{
    console.log('DOM loaded, waiting for 3Dmol.js...');
    waitFor3Dmol();
}});

// Also try initializing if window is already loaded
if (document.readyState === 'complete') {{
    console.log('Document already loaded, initializing...');
    waitFor3Dmol();
}}
</script>
<!-- END custom block -->
"""
    
    new_html = html.replace('<body>', '<body>' + inject_block)
    final_file = output_file
    with open(final_file, 'w', encoding='utf-8') as f:
        f.write(new_html)
    print(f"Top compounds dashboard saved to {final_file} (with embedded 3D viewers and JS)")
    
    try:
        os.remove(fig_file)
    except Exception:
        pass


# ------------------ Pipeline runner ------------------

def run_structural_alert_filter(sdf_file: str, alerts_list: List[str], output_file: Optional[str] = None, top_n: int = 50, protein_pdb: Optional[str] = None) -> None:
    df = load_molecules_from_sdf(sdf_file)
    df = calculate_physicochemical_properties(df)
    final_df = apply_structural_alerts(df, alerts_list)
    final_df = calculate_compound_scores(final_df)
    if output_file is None:
        base = Path(sdf_file).stem
        output_file = f"{base}_enhanced_analysis.sdf"
    save_results(final_df, output_file)
    passing_df = filter_passing_molecules(final_df)
    if len(passing_df) > 0:
        save_csv = f"{Path(output_file).stem}_passing_all_filters.csv"
        passing_df.to_csv(save_csv, index=False)
        vprint(f"Saved passing-all-filters CSV to {save_csv}")
    else:
        vprint("No molecules pass all filters!")
    output_prefix = Path(output_file).stem
    create_comprehensive_visualizations(final_df, output_prefix, protein_pdb=protein_pdb)
    export_top_compounds(final_df, n_compounds=top_n, output_prefix=output_prefix, suffix='')
    if len(passing_df) > 0:
        export_top_compounds(passing_df, n_compounds=int(top_n), output_prefix=output_prefix, suffix='_passing')
    print_analysis_summary(final_df)

def print_analysis_summary(df: pd.DataFrame) -> None:
    vprint("="*60); vprint("ANALYSIS SUMMARY"); vprint("="*60)
    vprint(f"Total molecules analyzed: {len(df)}")
    if 'Affinity_Numeric' in df.columns:
        a = df['Affinity_Numeric'].dropna()
        if len(a)>0:
            vprint(f"Affinity range: {a.min():.2f} to {a.max():.2f}; mean {a.mean():.2f}")
    if 'Lipinski_Compliant' in df.columns:
        lp = int(df['Lipinski_Compliant'].sum()); vprint(f"Lipinski-compliant molecules: {lp} ({lp/len(df)*100:.1f}%)")
    filter_cols = [c for c in df.columns if c.endswith('_pass_filter')]
    if filter_cols:
        all_pass = int(df[filter_cols].all(axis=1).sum()); vprint(f"Molecules passing all filters: {all_pass} ({all_pass/len(df)*100:.1f}%)")
    if 'Composite_Score' in df.columns and len(df)>0:
        top_idx = df['Composite_Score'].idxmax(); vprint(f"Top compound: {df.loc[top_idx,'name']} (score: {df.loc[top_idx,'Composite_Score']:.3f})")
    vprint("="*60)

# -------------- CLI --------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced molecular docking analysis with 3D viewers and interactive HTML", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("input", help="Input SDF file with docked molecules")
    parser.add_argument("-o","--output", help="Output SDF file (default: <input>_enhanced_analysis.sdf)")
    parser.add_argument("--top-n", type=int, default=10, help="Number of top compounds to export separately")
    parser.add_argument("--protein", help="Optional protein PDB file to include in 3D viewers when Dock is toggled")
    parser.add_argument("--list-alerts", action="store_true", help="List available alert filters and exit")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging for debugging")
    args = parser.parse_args()
    verbose = args.verbose
    if args.list_alerts:
        alerts = get_available_alerts(); vprint("Available alerts:"); vprint('\\n'.join(alerts)); sys.exit(0)
    if not os.path.isfile(args.input):
        print(f"Input file {args.input} not found"); sys.exit(1)
    if args.output is None:
        base = Path(args.input).stem; args.output = f"{base}_enhanced_analysis.sdf"
    alerts_list = get_available_alerts()
    vprint(f"Using {len(alerts_list)} alerts: {alerts_list}")
    try:
        run_structural_alert_filter(args.input, alerts_list, output_file=args.output, top_n=args.top_n, protein_pdb=args.protein)
        vprint("\nDone")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
