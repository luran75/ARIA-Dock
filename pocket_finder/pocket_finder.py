"""
Pocket Finder Module
====================

Uses pyKVFinder to detect binding pockets in protein structures.
Provides ranking based on volume, area, depth, and hydrophobicity.

Author: Louis (with Claude's help)
"""

import os
import tempfile
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import shutil

# Try importing pyKVFinder
try:
    import pyKVFinder
    PYKVFINDER_AVAILABLE = True
except ImportError:
    PYKVFINDER_AVAILABLE = False
    print("[WARNING] pyKVFinder not available. Install with: pip install pyKVFinder")


@dataclass
class Pocket:
    """Represents a detected binding pocket."""
    id: str                    # e.g., "KAA", "KAB", etc.
    name: str                  # e.g., "Pocket 1", "Pocket 2"
    volume: float              # in Å³
    area: float                # in Å²
    depth: float               # average depth in Å
    hydrophobicity: float      # hydrophobicity score
    residues: List[str]        # list of residues lining the pocket
    pdb_file: Optional[str]    # path to pocket PDB file
    color: str                 # hex color for visualization
    rank_score: float = 0.0    # computed ranking score


class PocketFinder:
    """
    Finds and ranks binding pockets in protein structures using pyKVFinder.
    """
    
    # Colors for different pockets (up to 10)
    POCKET_COLORS = [
        "#FF6B6B",  # Red
        "#4ECDC4",  # Teal
        "#45B7D1",  # Blue
        "#96CEB4",  # Green
        "#FFEAA7",  # Yellow
        "#DDA0DD",  # Plum
        "#98D8C8",  # Mint
        "#F7DC6F",  # Gold
        "#BB8FCE",  # Purple
        "#85C1E9",  # Light Blue
    ]
    
    def __init__(self, work_dir: str = None):
        """Initialize pocket finder with a working directory."""
        if work_dir:
            self.work_dir = Path(work_dir)
        else:
            self.work_dir = Path(tempfile.mkdtemp(prefix="pockets_"))
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
    def detect_pockets(self, pdb_file: str, min_volume: float = 100.0) -> Dict:
        """
        Detect binding pockets in a protein structure.
        
        Args:
            pdb_file: Path to the protein PDB file
            min_volume: Minimum pocket volume in Å³ (default 100)
        
        Returns:
            Dictionary with success status and list of Pocket objects
        """
        if not PYKVFINDER_AVAILABLE:
            return {
                "success": False,
                "error": "pyKVFinder not installed. Run: pip install pyKVFinder",
                "pockets": []
            }
        
        if not os.path.exists(pdb_file):
            return {
                "success": False,
                "error": f"PDB file not found: {pdb_file}",
                "pockets": []
            }
        
        print(f"[POCKET] Detecting pockets in: {pdb_file}")
        
        try:
            # Run pyKVFinder workflow
            result = pyKVFinder.run_workflow(
                pdb_file,
                include_depth=True,
                include_hydropathy=True,
                hydrophobicity_scale='EisenbergWeiss',
                verbose=False
            )
            
            if result is None:
                return {
                    "success": True,
                    "pockets": [],
                    "cavity_pdb": None,
                    "message": "No cavities found in this structure"
                }
            
            # Get the base name for output files
            pdb_name = Path(pdb_file).stem
            output_prefix = self.work_dir / pdb_name
            
            # Export results
            cavity_pdb = str(output_prefix) + "_cavities.pdb"
            results_toml = str(output_prefix) + "_results.toml"
            
            result.export_all(
                fn=results_toml,
                output=cavity_pdb,
                include_frequencies_pdf=False  # Skip PDF generation for speed
            )
            
            print(f"[POCKET] Cavities exported to: {cavity_pdb}")
            
            # Parse pockets from results
            pockets = self._parse_pockets(result, cavity_pdb, min_volume)
            
            # If all pockets were filtered out, return all of them anyway
            if len(pockets) == 0 and result.ncav > 0:
                print(f"[POCKET] All pockets below {min_volume} Å³, returning all...")
                pockets = self._parse_pockets(result, cavity_pdb, min_volume=0)
            
            # Calculate ranking scores
            self._calculate_rankings(pockets)
            
            # Sort by rank score (highest first)
            pockets.sort(key=lambda p: p.rank_score, reverse=True)
            
            # Extract individual pocket PDB files
            self._extract_pocket_pdbs(cavity_pdb, pockets)
            
            print(f"[POCKET] Found {len(pockets)} pockets")
            for p in pockets:
                print(f"  - {p.name}: Vol={p.volume:.1f}Å³, Area={p.area:.1f}Å², "
                      f"Depth={p.depth:.2f}Å, Score={p.rank_score:.2f}")
            
            return {
                "success": True,
                "pockets": pockets,
                "cavity_pdb": cavity_pdb,
                "message": f"Found {len(pockets)} binding pockets"
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e),
                "pockets": []
            }
    
    def _parse_pockets(self, result, cavity_pdb: str, min_volume: float) -> List[Pocket]:
        """Parse pocket information from pyKVFinder results."""
        pockets = []
        
        # Get cavity names (e.g., "KAA", "KAB", etc.)
        cavity_names = list(result.volume.keys())
        
        for i, cav_id in enumerate(cavity_names):
            volume = result.volume.get(cav_id, 0)
            
            # Skip pockets below minimum volume
            if volume < min_volume:
                continue
            
            area = result.area.get(cav_id, 0)
            
            # Get depth (average if available)
            depth = 0.0
            if hasattr(result, 'avg_depth') and result.avg_depth:
                depth = result.avg_depth.get(cav_id, 0)
            elif hasattr(result, 'max_depth') and result.max_depth:
                depth = result.max_depth.get(cav_id, 0)
            
            # Get hydrophobicity (average if available)
            hydrophobicity = 0.0
            if hasattr(result, 'avg_hydropathy') and result.avg_hydropathy:
                hydrophobicity = result.avg_hydropathy.get(cav_id, 0)
            
            # Get residues lining the pocket
            residues = []
            if hasattr(result, 'residues') and result.residues:
                residues = result.residues.get(cav_id, [])
            
            # Assign color
            color = self.POCKET_COLORS[i % len(self.POCKET_COLORS)]
            
            pocket = Pocket(
                id=cav_id,
                name=f"Pocket {i + 1}",
                volume=volume,
                area=area,
                depth=depth,
                hydrophobicity=hydrophobicity,
                residues=residues,
                pdb_file=None,  # Will be set later
                color=color
            )
            
            pockets.append(pocket)
        
        return pockets
    
    def _calculate_rankings(self, pockets: List[Pocket]):
        """Calculate ranking scores for pockets based on their properties."""
        if not pockets:
            return
        
        # Get min/max for normalization
        volumes = [p.volume for p in pockets]
        areas = [p.area for p in pockets]
        depths = [p.depth for p in pockets]
        hydros = [p.hydrophobicity for p in pockets]
        
        def normalize(value, values):
            """Normalize value to 0-1 range."""
            min_val, max_val = min(values), max(values)
            if max_val == min_val:
                return 0.5
            return (value - min_val) / (max_val - min_val)
        
        # Calculate weighted scores
        # Weights: Volume (0.35), Area (0.30), Depth (0.25), Hydrophobicity (0.10)
        for pocket in pockets:
            norm_volume = normalize(pocket.volume, volumes)
            norm_area = normalize(pocket.area, areas)
            norm_depth = normalize(pocket.depth, depths)
            norm_hydro = normalize(pocket.hydrophobicity, hydros)
            
            pocket.rank_score = (
                0.35 * norm_volume +
                0.30 * norm_area +
                0.25 * norm_depth +
                0.10 * norm_hydro
            ) * 100  # Scale to 0-100
    
    def _extract_pocket_pdbs(self, cavity_pdb: str, pockets: List[Pocket]):
        """Extract individual pocket PDB files from the combined cavity PDB."""
        if not os.path.exists(cavity_pdb):
            return
        
        try:
            # Read the cavity PDB
            with open(cavity_pdb, 'r') as f:
                lines = f.readlines()
            
            # Group atoms by residue name (pocket ID)
            pocket_atoms = {}
            for line in lines:
                if line.startswith(('ATOM', 'HETATM')):
                    res_name = line[17:20].strip()
                    if res_name not in pocket_atoms:
                        pocket_atoms[res_name] = []
                    pocket_atoms[res_name].append(line)
            
            # Save individual pocket PDBs
            for pocket in pockets:
                if pocket.id in pocket_atoms:
                    pocket_file = self.work_dir / f"pocket_{pocket.id}.pdb"
                    with open(pocket_file, 'w') as f:
                        f.write(f"HEADER    POCKET {pocket.id}\n")
                        f.write(f"REMARK    Volume: {pocket.volume:.1f} A^3\n")
                        f.write(f"REMARK    Area: {pocket.area:.1f} A^2\n")
                        f.writelines(pocket_atoms[pocket.id])
                        f.write("END\n")
                    pocket.pdb_file = str(pocket_file)
                    print(f"[POCKET] Saved {pocket.name} to {pocket_file}")
                    
        except Exception as e:
            print(f"[POCKET] Error extracting pocket PDBs: {e}")
    
    def get_pocket_by_id(self, pockets: List[Pocket], pocket_id: str) -> Optional[Pocket]:
        """Get a pocket by its ID (e.g., 'KAA')."""
        for pocket in pockets:
            if pocket.id == pocket_id:
                return pocket
        return None


def create_pocket_table_html(pockets: List[Pocket]) -> str:
    """Create an HTML table showing pocket rankings and properties."""
    if not pockets:
        return """
        <div style="padding: 20px; text-align: center; color: #94a3b8;">
            No pockets detected in this structure.
        </div>
        """
    
    rows = ""
    for i, pocket in enumerate(pockets):
        # Color indicator
        color_dot = f'<span style="display:inline-block; width:12px; height:12px; border-radius:50%; background:{pocket.color}; margin-right:8px;"></span>'
        
        # Rank badge
        if i == 0:
            rank_badge = '<span style="background:#22c55e; color:white; padding:2px 8px; border-radius:10px; font-size:11px;">Best</span>'
        else:
            rank_badge = f'<span style="background:#334155; color:#94a3b8; padding:2px 8px; border-radius:10px; font-size:11px;">#{i+1}</span>'
        
        rows += f"""
        <tr style="border-bottom: 1px solid #334155;">
            <td style="padding: 10px; color: #f1f5f9;">{color_dot}{pocket.name}</td>
            <td style="padding: 10px; text-align: center;">{rank_badge}</td>
            <td style="padding: 10px; text-align: right; color: #60a5fa;">{pocket.volume:.0f} Å³</td>
            <td style="padding: 10px; text-align: right; color: #4ade80;">{pocket.area:.0f} Å²</td>
            <td style="padding: 10px; text-align: right; color: #fbbf24;">{pocket.depth:.1f} Å</td>
            <td style="padding: 10px; text-align: right; color: #c084fc;">{pocket.hydrophobicity:.2f}</td>
            <td style="padding: 10px; text-align: right; color: #f1f5f9; font-weight: bold;">{pocket.rank_score:.1f}</td>
        </tr>
        """
    
    return f"""
    <div style="background: #0f172a; border-radius: 12px; padding: 16px; margin-top: 16px;">
        <h4 style="color: #f1f5f9; margin: 0 0 12px 0; font-size: 14px;">
            🎯 Detected Binding Pockets
        </h4>
        <div style="overflow-x: auto;">
            <table style="width: 100%; border-collapse: collapse; font-size: 13px;">
                <thead>
                    <tr style="border-bottom: 2px solid #334155;">
                        <th style="padding: 8px; text-align: left; color: #94a3b8;">Pocket</th>
                        <th style="padding: 8px; text-align: center; color: #94a3b8;">Rank</th>
                        <th style="padding: 8px; text-align: right; color: #94a3b8;">Volume</th>
                        <th style="padding: 8px; text-align: right; color: #94a3b8;">Area</th>
                        <th style="padding: 8px; text-align: right; color: #94a3b8;">Depth</th>
                        <th style="padding: 8px; text-align: right; color: #94a3b8;">Hydro.</th>
                        <th style="padding: 8px; text-align: right; color: #94a3b8;">Score</th>
                    </tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>
        </div>
        <div style="margin-top: 12px; font-size: 11px; color: #64748b;">
            💡 Click on a pocket in the 3D viewer to select it for docking
        </div>
    </div>
    """



# ============================================================================
# 3D Dashboard  —  Helper functions
# ============================================================================

def _parse_cavity_coordinates(cavity_pdb: str, pockets: List[Pocket]) -> Dict[str, List[Tuple[float, float, float]]]:
    """
    Parse atom XYZ coordinates from the pyKVFinder cavity PDB, grouped by pocket ID.

    pyKVFinder encodes the cavity ID as the residue name (e.g. KAA, KAB …)
    at columns 17–20 of every ATOM / HETATM record.
    """
    coords: Dict[str, List[Tuple[float, float, float]]] = {}
    pocket_ids = {p.id for p in pockets}

    if not cavity_pdb or not os.path.exists(cavity_pdb):
        return coords

    try:
        with open(cavity_pdb, "r") as fh:
            for line in fh:
                if line.startswith(("ATOM", "HETATM")) and len(line) >= 54:
                    res_name = line[17:20].strip()
                    if res_name in pocket_ids:
                        try:
                            x = float(line[30:38])
                            y = float(line[38:46])
                            z = float(line[46:54])
                            coords.setdefault(res_name, []).append((x, y, z))
                        except ValueError:
                            pass
    except Exception as exc:
        print(f"[POCKET] Warning – could not parse cavity coordinates: {exc}")

    return coords


def _build_pocket_sphere_pdb(coords: List[Tuple[float, float, float]]) -> str:
    """
    Build a minimal PDB string whose atoms sit at every cavity grid point.
    These are loaded into 3Dmol.js as a separate model and styled as spheres,
    which is much faster than adding individual shape objects.
    """
    lines = ["REMARK POCKET GRID POINTS"]
    for i, (x, y, z) in enumerate(coords, 1):
        n = (i - 1) % 99999 + 1
        lines.append(
            f"HETATM{n:5d}  C   POK A   1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
        )
    lines.append("END")
    return "\n".join(lines)


# ============================================================================
# 3D Dashboard  —  Main builder
# ============================================================================

def create_pocket_dashboard_html(
    pockets: List[Pocket],
    protein_pdb_path: str,
    cavity_pdb_path: str,
    protein_name: str = "",
) -> str:
    """
    Create a fully self-contained HTML dashboard for visualising binding pockets.

    The file embeds the protein PDB, per-pocket grid-point PDBs, and all pocket
    metadata as JSON so no web server or external files are required.

    Args:
        pockets:           List of Pocket objects returned by detect_pockets().
        protein_pdb_path:  Path to the original protein PDB file.
        cavity_pdb_path:   Path to the combined cavity PDB written by pyKVFinder.
        protein_name:      Display name shown in the dashboard header.
                           Defaults to the stem of protein_pdb_path.

    Returns:
        Complete HTML string ready to be written to disk.
    """
    import json

    # ── protein PDB ──────────────────────────────────────────────────────────
    try:
        with open(protein_pdb_path, "r") as fh:
            protein_pdb_content = fh.read()
    except Exception as exc:
        protein_pdb_content = f"REMARK Error reading protein: {exc}\nEND"

    if not protein_name:
        protein_name = Path(protein_pdb_path).stem.upper()

    # ── cavity coordinates ───────────────────────────────────────────────────
    cavity_coords = _parse_cavity_coordinates(cavity_pdb_path, pockets)

    # ── pocket data for JavaScript ───────────────────────────────────────────
    pockets_js_data = []
    all_lining_residues: set = set()

    for i, pocket in enumerate(pockets):
        coords = cavity_coords.get(pocket.id, [])

        # centroid of grid points
        if coords:
            cx = sum(c[0] for c in coords) / len(coords)
            cy = sum(c[1] for c in coords) / len(coords)
            cz = sum(c[2] for c in coords) / len(coords)
        else:
            cx = cy = cz = 0.0

        # per-pocket PDB (loaded into 3Dmol as a model)
        pocket_pdb_str = _build_pocket_sphere_pdb(coords) if coords else "REMARK EMPTY\nEND"

        # residues
        formatted_residues = []
        for res in pocket.residues:
            if isinstance(res, (list, tuple)) and len(res) >= 3:
                # pyKVFinder residues format is [resnum, chain, resname]
                # e.g. ['14', 'E', 'SER']  — resnum is index 0, chain is index 1.
                # Previous code had these swapped, which caused the JS 3D lookup
                # to use residue numbers as chain IDs and fail silently on every residue.
                resnum  = str(res[0]).strip()
                chain   = str(res[1]).strip()
                resname = str(res[2]).strip()
                formatted_residues.append(
                    {"chain": chain, "resnum": resnum, "resname": resname}
                )
                all_lining_residues.add(f"{chain}:{resnum}:{resname}")
            elif isinstance(res, str):
                formatted_residues.append({"chain": "?", "resnum": "?", "resname": res.strip()})
                all_lining_residues.add(res.strip())

        pockets_js_data.append(
            {
                "id":             pocket.id,
                "name":           pocket.name,
                "color":          pocket.color,
                "volume":         round(pocket.volume, 1),
                "area":           round(pocket.area, 1),
                "depth":          round(pocket.depth, 2),
                "hydrophobicity": round(pocket.hydrophobicity, 3),
                "score":          round(pocket.rank_score, 1),
                "rank":           i + 1,
                "is_best":        i == 0,
                "centroid":       [round(cx, 2), round(cy, 2), round(cz, 2)],
                "n_points":       len(coords),
                "pdb":            pocket_pdb_str,
                "residues":       formatted_residues,
            }
        )

    # JSON-encode everything (handles all escaping automatically)
    pockets_json        = json.dumps(pockets_js_data)
    protein_pdb_json    = json.dumps(protein_pdb_content)
    n_pockets           = str(len(pockets))
    n_lining_residues   = str(len(all_lining_residues))

    # ── inject into template ─────────────────────────────────────────────────
    template = _get_dashboard_template()
    html = (
        template
        .replace("MARKER_PROTEIN_NAME",     protein_name)
        .replace("MARKER_N_POCKETS",        n_pockets)
        .replace("MARKER_N_RESIDUES",       n_lining_residues)
        .replace("MARKER_PROTEIN_PDB",      protein_pdb_json)
        .replace("MARKER_POCKETS_JSON",     pockets_json)
    )
    return html


# ============================================================================
# 3D Dashboard  —  HTML / CSS / JS template
# ============================================================================

def _get_dashboard_template() -> str:
    """Return the self-contained HTML template. Markers replaced by caller."""
    # NOTE: raw string — backslashes are literal, safe to embed JS.
    return r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PocketFinder — MARKER_PROTEIN_NAME</title>

<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Oxanium:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&family=DM+Sans:wght@400;500&display=swap" rel="stylesheet">

<script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.7.1/jquery.min.js"></script>
<script src="https://3Dmol.csb.pitt.edu/build/3Dmol-min.js"></script>

<style>
/* ── reset & tokens ─────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg:           #06101e;
  --panel:        #0a1828;
  --panel-hi:     #0d1f32;
  --border:       #162e48;
  --border-hi:    #1e4268;
  --text:         #c8ddf2;
  --text-dim:     #4a7096;
  --text-bright:  #eaf4ff;
  --accent:       #00ccff;
  --accent-glow:  rgba(0,204,255,0.15);
  --success:      #22c55e;
  --blue:         #60a5fa;
  --green:        #4ade80;
  --amber:        #fbbf24;
  --purple:       #c084fc;
  --font-display: 'Oxanium', sans-serif;
  --font-mono:    'JetBrains Mono', 'Courier New', monospace;
  --font-body:    'DM Sans', sans-serif;
  --radius:       8px;
  --sidebar-w:    390px;
}

html, body {
  height: 100%;
  overflow: hidden;
  background: var(--bg);
  color: var(--text);
  font-family: var(--font-body);
  font-size: 13px;
}

/* subtle grid overlay on the full page */
body::before {
  content: '';
  position: fixed; inset: 0; z-index: 0; pointer-events: none;
  background-image:
    linear-gradient(rgba(0,204,255,0.02) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,204,255,0.02) 1px, transparent 1px);
  background-size: 52px 52px;
}

/* ── layout ─────────────────────────────────────────────── */
#app {
  position: relative; z-index: 1;
  display: flex;
  height: 100vh;
  width: 100vw;
}

/* ── viewer (left, main area) ───────────────────────────── */
#viewer-wrap {
  flex: 1;
  position: relative;
  overflow: hidden;
  background: var(--bg);
}
#viewer {
  width: 100%;
  height: 100%;
}

/* ── loading overlay ────────────────────────────────────── */
#loading-overlay {
  position: absolute; inset: 0;
  background: rgba(6,16,30,0.90);
  display: flex; flex-direction: column;
  align-items: center; justify-content: center;
  gap: 18px; z-index: 200;
  backdrop-filter: blur(3px);
  transition: opacity 0.45s ease;
}
.spinner {
  width: 38px; height: 38px;
  border: 2px solid var(--border-hi);
  border-top-color: var(--accent);
  border-radius: 50%;
  animation: spin 0.85s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
.load-msg {
  font-family: var(--font-display);
  font-size: 16px; letter-spacing: 3px;
  color: var(--text-dim);
}

/* ── viewer toolbar (overlay, top-left) ─────────────────── */
#toolbar {
  position: absolute; top: 14px; left: 14px;
  z-index: 100;
  display: flex; gap: 7px; flex-wrap: wrap;
}
.tool-btn {
  font-family: var(--font-mono); font-size: 11px;
  letter-spacing: 0.06em;
  color: var(--text-dim);
  background: rgba(6,16,30,0.88);
  border: 1px solid var(--border-hi);
  border-radius: 6px; padding: 6px 13px;
  cursor: pointer; backdrop-filter: blur(6px);
  display: flex; align-items: center; gap: 6px;
  transition: color 0.15s, border-color 0.15s, background 0.15s;
}
.tool-btn:hover {
  color: var(--accent);
  border-color: rgba(0,204,255,0.40);
  background: rgba(0,204,255,0.07);
}
.tool-btn.active {
  color: var(--accent);
  border-color: rgba(0,204,255,0.50);
  background: rgba(0,204,255,0.09);
}
.tool-btn svg { width: 12px; height: 12px; flex-shrink: 0; }

/* watermark */
#viewer-badge {
  position: absolute; bottom: 12px; right: 12px;
  font-family: var(--font-display); font-size: 10px;
  letter-spacing: 2px;
  color: rgba(255,255,255,0.10);
  pointer-events: none; user-select: none;
}

/* ── sidebar (right) ────────────────────────────────────── */
#sidebar {
  width: var(--sidebar-w); min-width: var(--sidebar-w);
  height: 100vh;
  display: flex; flex-direction: column;
  background: var(--panel);
  border-left: 1px solid var(--border);
  overflow: hidden;
}

/* ── sidebar header ─────────────────────────────────────── */
#header {
  flex-shrink: 0;
  padding: 18px 18px 15px;
  border-bottom: 1px solid var(--border);
  background: linear-gradient(160deg, #091525 0%, var(--panel) 100%);
}
.eyebrow {
  font-family: var(--font-display); font-size: 10px;
  letter-spacing: 4px; color: var(--accent);
  margin-bottom: 5px;
}
.protein-name {
  font-family: var(--font-display); font-size: 28px;
  letter-spacing: 2px; color: var(--text-bright);
  line-height: 1; word-break: break-all;
  margin-bottom: 12px;
}
.header-chips {
  display: flex; gap: 8px; flex-wrap: wrap;
}
.chip {
  font-family: var(--font-mono); font-size: 10px;
  color: var(--text-dim);
  background: rgba(0,0,0,0.3);
  border: 1px solid var(--border-hi);
  border-radius: 4px; padding: 3px 9px;
  display: flex; align-items: center; gap: 5px;
}
.chip strong { color: var(--text); }

/* ── controls ───────────────────────────────────────────── */
#controls {
  flex-shrink: 0;
  padding: 13px 18px;
  border-bottom: 1px solid var(--border);
}
.section-title {
  font-family: var(--font-display); font-size: 10px;
  letter-spacing: 3.5px; color: var(--text-dim);
  margin-bottom: 10px;
}
.ctrl-row {
  display: flex; align-items: center;
  gap: 10px; margin-bottom: 9px;
}
.ctrl-lbl {
  font-family: var(--font-mono); font-size: 10px;
  color: var(--text-dim); width: 90px; flex-shrink: 0;
  letter-spacing: 0.04em;
}
.ctrl-val {
  font-family: var(--font-mono); font-size: 11px;
  color: var(--accent); width: 38px;
  text-align: right; flex-shrink: 0;
}
input[type="range"] {
  flex: 1; height: 3px;
  -webkit-appearance: none;
  background: var(--border-hi); border-radius: 2px;
  outline: none; cursor: pointer;
}
input[type="range"]::-webkit-slider-thumb {
  -webkit-appearance: none;
  width: 13px; height: 13px; border-radius: 50%;
  background: var(--accent);
  box-shadow: 0 0 6px rgba(0,204,255,0.5);
  cursor: pointer;
}
input[type="range"]::-moz-range-thumb {
  width: 13px; height: 13px; border: none;
  border-radius: 50%; background: var(--accent);
  cursor: pointer;
}
.ctrl-buttons {
  display: flex; gap: 7px; margin-top: 10px;
}
.btn {
  flex: 1; padding: 6px 0;
  border: 1px solid var(--border-hi);
  border-radius: var(--radius);
  background: transparent; color: var(--text-dim);
  font-family: var(--font-mono); font-size: 10px;
  letter-spacing: 1px; cursor: pointer;
  transition: all 0.14s; text-align: center;
}
.btn:hover {
  border-color: var(--accent);
  color: var(--accent);
  background: var(--accent-glow);
}
.btn.danger        { color: #f87171; border-color: #3f1212; }
.btn.danger:hover  { color: #ff9090; border-color: #7a2020; background: rgba(248,113,113,0.07); }

/* "Show pocket residues in 3D" button */
.btn-residues {
  width: 100%; margin-top: 9px; padding: 8px 12px;
  display: flex; align-items: center; justify-content: center;
  gap: 7px; letter-spacing: 0.06em;
  border: 1px dashed var(--border-hi);
  color: var(--text-dim);
  transition: all 0.18s;
}
.btn-residues:hover {
  border-style: solid;
  border-color: var(--accent);
  color: var(--accent);
  background: var(--accent-glow);
}
.btn-residues.active {
  border-style: solid;
  border-color: var(--success);
  color: var(--success);
  background: rgba(34, 197, 94, 0.07);
}
.btn-residues:disabled,
.btn-residues.waiting {
  opacity: 0.45; cursor: default;
}
.btn-residues.waiting:hover {
  border-color: var(--border-hi);
  color: var(--text-dim);
  background: transparent;
}
#res-btn-hint {
  font-family: var(--font-mono); font-size: 10px;
  color: var(--text-dim); font-style: italic;
  text-align: center; margin-top: 5px; letter-spacing: 0.03em;
  min-height: 14px;
  transition: opacity 0.2s;
}

/* deselect hint shown inside selected card */
.deselect-hint {
  font-family: var(--font-mono); font-size: 9px;
  color: var(--text-dim); font-style: italic;
  letter-spacing: 0.04em;
  margin-left: 6px;
  opacity: 0;
  transition: opacity 0.3s 0.2s;
}

/* surface status indicator */
#surf-status {
  font-family: var(--font-mono); font-size: 10px;
  color: var(--text-dim); margin-top: 8px;
  display: flex; align-items: center; gap: 6px;
  min-height: 16px;
}
.surf-dot {
  width: 7px; height: 7px; border-radius: 50%;
  background: var(--accent); flex-shrink: 0;
  animation: pulse 1.5s ease-in-out infinite;
}
@keyframes pulse {
  0%, 100% { opacity: 1; }
  50%       { opacity: 0.3; }
}
.surf-dot.done    { background: var(--success); animation: none; }
.surf-dot.off     { background: var(--border-hi); animation: none; }

/* ── pocket list (scrollable) ───────────────────────────── */
#pocket-section {
  flex: 1; overflow-y: auto; min-height: 0;
  scrollbar-width: thin;
  scrollbar-color: var(--border-hi) transparent;
}
#pocket-section::-webkit-scrollbar { width: 4px; }
#pocket-section::-webkit-scrollbar-thumb { background: var(--border-hi); border-radius: 2px; }

.pocket-section-title {
  font-family: var(--font-display); font-size: 10px;
  letter-spacing: 3.5px; color: var(--text-dim);
  padding: 13px 18px 7px; flex-shrink: 0;
}

#pocket-list { padding: 4px 12px 12px; }

/* ── pocket card ────────────────────────────────────────── */
.pocket-card {
  margin-bottom: 8px;
  border-radius: var(--radius);
  border: 1px solid var(--border);
  border-left: 3px solid var(--card-color, #fff);
  background: var(--panel-hi);
  cursor: pointer;
  transition: background 0.15s, box-shadow 0.2s, border-color 0.15s;
  animation: cardIn 0.35s ease both;
  overflow: hidden;
}
@keyframes cardIn {
  from { opacity: 0; transform: translateX(16px); }
  to   { opacity: 1; transform: translateX(0); }
}
.pocket-card:hover {
  background: #0e2238;
  border-color: var(--border-hi);
  border-left-color: var(--card-color, #fff);
}
.pocket-card.selected {
  background: #0c2040;
  border-color: var(--card-color, var(--accent));
  box-shadow: 0 0 18px -5px var(--card-color, rgba(0,204,255,0.3));
}

.card-inner { padding: 11px 13px; }

/* card top row */
.card-top {
  display: flex; justify-content: space-between;
  align-items: center; margin-bottom: 8px;
}
.card-name-row {
  display: flex; align-items: center; gap: 8px;
}
.pocket-num {
  font-family: var(--font-display); font-size: 17px;
  color: var(--card-color, var(--text-bright)); line-height: 1;
}
.pocket-id {
  font-family: var(--font-mono); font-size: 10px;
  color: var(--text-dim); letter-spacing: 0.08em;
}
.best-badge {
  font-family: var(--font-display); font-size: 9px;
  letter-spacing: 1.5px;
  background: var(--success); color: #fff;
  padding: 2px 6px; border-radius: 3px;
}
/* toggle switch */
.toggle-wrap {
  display: flex; align-items: center;
  gap: 6px; cursor: pointer; user-select: none;
  flex-shrink: 0;
}
.toggle-wrap input[type="checkbox"] { display: none; }
.toggle-pill {
  width: 30px; height: 16px; border-radius: 8px;
  background: var(--border-hi); position: relative;
  transition: background 0.2s; flex-shrink: 0;
}
.toggle-pill::after {
  content: ''; position: absolute;
  top: 2px; left: 2px;
  width: 12px; height: 12px; border-radius: 50%;
  background: var(--text-dim);
  transition: transform 0.2s, background 0.2s;
}
.toggle-wrap.on .toggle-pill { background: var(--card-color, var(--accent)); }
.toggle-wrap.on .toggle-pill::after { transform: translateX(14px); background: #fff; }
.toggle-lbl {
  font-family: var(--font-mono); font-size: 10px;
  color: var(--text-dim); letter-spacing: 0.04em;
}

/* stats grid */
.card-stats {
  display: grid; grid-template-columns: 1fr 1fr;
  gap: 4px 14px; margin-bottom: 8px;
}
.stat-row {
  display: flex; justify-content: space-between;
  align-items: baseline;
}
.stat-key {
  font-family: var(--font-mono); font-size: 9px;
  color: var(--text-dim); text-transform: uppercase;
  letter-spacing: 0.8px;
}
.stat-val { font-family: var(--font-mono); font-size: 12px; font-weight: 500; }
.c-blue   { color: var(--blue); }
.c-green  { color: var(--green); }
.c-amber  { color: var(--amber); }
.c-purple { color: var(--purple); }

/* score bar */
.score-row {
  display: flex; align-items: center; gap: 8px;
  margin-bottom: 8px;
}
.score-key {
  font-family: var(--font-mono); font-size: 9px;
  letter-spacing: 1.5px; color: var(--text-dim);
  width: 38px; flex-shrink: 0;
}
.score-track {
  flex: 1; height: 4px;
  background: var(--border-hi); border-radius: 2px; overflow: hidden;
}
.score-fill {
  height: 100%; border-radius: 2px;
  width: 0%; transition: width 1.1s cubic-bezier(0.22, 1, 0.36, 1);
}
.score-num {
  font-family: var(--font-display); font-size: 15px;
  color: var(--text-bright); width: 32px;
  text-align: right; flex-shrink: 0;
}

/* inline residues (collapsible) */
.res-details {
  font-family: var(--font-mono); font-size: 10px;
}
.res-details summary {
  color: var(--text-dim); cursor: pointer; user-select: none;
  padding: 2px 0; letter-spacing: 0.05em;
  list-style: none; transition: color 0.14s;
}
.res-details summary::-webkit-details-marker { display: none; }
.res-details summary::before { content: '\25B6  '; font-size: 8px; }
.res-details[open] summary::before { content: '\25BC  '; }
.res-details summary:hover { color: var(--text); }
.res-grid {
  display: flex; flex-wrap: wrap; gap: 4px; padding-top: 8px;
}
.res-pill {
  font-size: 10px; padding: 2px 7px;
  border-radius: 4px; border: 1px solid var(--border-hi);
  color: var(--text); background: var(--panel-hi);
  letter-spacing: 0.4px;
}
.res-pill .ch { color: var(--text-dim); margin-right: 2px; }
.res-none {
  font-style: italic; color: var(--text-dim); font-size: 10px;
}

/* no-pockets state */
#no-pockets {
  display: flex; flex-direction: column;
  align-items: center; justify-content: center;
  gap: 10px; color: var(--text-dim);
  text-align: center; padding: 40px 20px;
}
.np-icon { font-size: 30px; opacity: 0.35; }

/* res-ready: subtle animated ring to hint the button is now usable */
.tool-btn.res-ready {
  border-color: rgba(0,204,255,0.35);
  color: var(--text);
  animation: readyRing 2s ease-in-out 3;
}
@keyframes readyRing {
  0%   { box-shadow: 0 0 0 0   rgba(0,204,255,0.50); }
  50%  { box-shadow: 0 0 0 4px rgba(0,204,255,0.20); }
  100% { box-shadow: 0 0 0 0   rgba(0,204,255,0.00); }
}

/* toast notification */
#toast {
  position: absolute; bottom: 40px; left: 50%; transform: translateX(-50%);
  background: rgba(6,16,30,0.92); border: 1px solid var(--border-hi);
  border-radius: 6px; padding: 8px 16px;
  font-family: var(--font-mono); font-size: 11px;
  color: var(--text); letter-spacing: 0.04em;
  backdrop-filter: blur(6px);
  opacity: 0; pointer-events: none;
  transition: opacity 0.25s ease;
  z-index: 300; white-space: nowrap;
}
#toast.show { opacity: 1; }
.pocket-card:nth-child(2)  { animation-delay: 0.09s; }
.pocket-card:nth-child(3)  { animation-delay: 0.14s; }
.pocket-card:nth-child(4)  { animation-delay: 0.19s; }
.pocket-card:nth-child(5)  { animation-delay: 0.24s; }
.pocket-card:nth-child(6)  { animation-delay: 0.29s; }
.pocket-card:nth-child(7)  { animation-delay: 0.34s; }
.pocket-card:nth-child(8)  { animation-delay: 0.39s; }
.pocket-card:nth-child(9)  { animation-delay: 0.44s; }
.pocket-card:nth-child(10) { animation-delay: 0.49s; }
</style>
</head>

<body>
<div id="app">

  <!-- ═══════════════════ VIEWER (left, main) ═══════════════════ -->
  <main id="viewer-wrap">
    <div id="viewer"></div>

    <!-- Loading overlay -->
    <div id="loading-overlay">
      <div class="spinner"></div>
      <div class="load-msg" id="loading-text">LOADING STRUCTURE</div>
    </div>

    <!-- Quick-action toolbar (overlay) -->
    <div id="toolbar">
      <button class="tool-btn" onclick="resetView()">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M3 12a9 9 0 1 0 18 0 9 9 0 0 0-18 0"/>
          <polyline points="3 12 3 3 12 3"/>
        </svg>Reset View
      </button>
      <button class="tool-btn" onclick="toggleAllPockets(true)">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/>
        </svg>Show All
      </button>
      <button class="tool-btn" onclick="toggleAllPockets(false)">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24"/>
          <line x1="1" y1="1" x2="23" y2="23"/>
        </svg>Hide All
      </button>
      <button class="tool-btn" id="surf-toggle-btn" onclick="onSurfaceToggle()">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="10"/>
          <path d="M12 2a14.5 14.5 0 0 0 0 20 14.5 14.5 0 0 0 0-20"/>
        </svg>Surface: OFF
      </button>
      <button class="tool-btn" id="res-stick-btn" onclick="toggleResidueSticks()" title="Show the lining residues of the selected pocket as sticks in the 3D viewer">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M12 2L2 7l10 5 10-5-10-5z"/>
          <path d="M2 17l10 5 10-5"/>
          <path d="M2 12l10 5 10-5"/>
        </svg><span id="res-stick-label">Residues</span>
      </button>
    </div>

    <!-- Watermark -->
    <div id="viewer-badge">POCKETFINDER · MARKER_PROTEIN_NAME</div>

    <!-- Toast notification -->
    <div id="toast"></div>
  </main>

  <!-- ═══════════════════ SIDEBAR (right) ═══════════════════════ -->
  <aside id="sidebar">

    <!-- Header -->
    <div id="header">
      <div class="eyebrow">POCKETFINDER</div>
      <div class="protein-name">MARKER_PROTEIN_NAME</div>
      <div class="header-chips">
        <span class="chip">Pockets&nbsp;<strong id="stat-pockets">MARKER_N_POCKETS</strong></span>
        <span class="chip">Lining res.&nbsp;<strong>MARKER_N_RESIDUES</strong></span>
      </div>
    </div>

    <!-- Controls -->
    <div id="controls">
      <div class="section-title">VISUALIZATION</div>

      <div class="ctrl-row">
        <span class="ctrl-lbl">Sphere radius</span>
        <input type="range" id="sphere-slider"
               min="0.3" max="2.0" step="0.05" value="0.85"
               oninput="onSphereRadius(this.value)">
        <span class="ctrl-val" id="sphere-val">0.85</span>
      </div>

      <div class="ctrl-row">
        <span class="ctrl-lbl">Surf. opacity</span>
        <input type="range" id="surf-slider"
               min="0" max="0.40" step="0.01" value="0"
               oninput="onSurfaceOpacity(this.value)">
        <span class="ctrl-val" id="surf-val">off</span>
      </div>

      <div class="ctrl-buttons">
        <button class="btn" onclick="resetView()">⌂ Reset</button>
        <button class="btn" onclick="toggleAllPockets(true)">Show all</button>
        <button class="btn" onclick="toggleAllPockets(false)">Hide all</button>
      </div>

      <!-- Residue sticks — lives in the sidebar so users can always find it -->
      <button class="btn btn-residues" id="res-stick-sidebar-btn"
              onclick="toggleResidueSticks()"
              title="Show the lining residues of the selected pocket as sticks in the 3D viewer">
        <span id="res-btn-icon">⬡</span>
        <span id="res-btn-text">Show pocket residues in 3D</span>
      </button>
      <div id="res-btn-hint">← select a pocket first</div>

      <div id="surf-status">
        <span class="surf-dot" id="surf-dot"></span>
        <span id="surf-msg">Computing surface…</span>
      </div>
    </div>

    <!-- Pocket list -->
    <div id="pocket-section">
      <div class="pocket-section-title">BINDING POCKETS</div>
      <div id="pocket-list"><!-- built by JS --></div>
    </div>

  </aside>

</div><!-- /app -->


<script>
// ══════════════════════════════════════════════════════════════
//  Injected data
// ══════════════════════════════════════════════════════════════
const PROTEIN_PDB = MARKER_PROTEIN_PDB;
const POCKETS     = MARKER_POCKETS_JSON;
const NOPOCKETS   = (POCKETS.length === 0);

// ══════════════════════════════════════════════════════════════
//  State
// ══════════════════════════════════════════════════════════════
let viewer            = null;
let proteinModel      = null;
let PROTEIN_MODEL_ID  = 0;   // protein is always the first model added
let pocketModels      = [];
let pocketVisible     = [];
let selectedIdx       = null;
let surfaceObj        = null;
let surfaceVisible    = false;
let surfaceReady      = false;
let sphereRadius      = 0.85;
let surfOpacity       = 0.0;
let showResidueSticks = false;  // whether lining residues are shown as sticks
// Generation counter: incremented whenever we start a new surface computation OR
// cancel an in-flight one.  Each async callback captures its own generation at
// dispatch time and aborts silently if the counter has since moved on.
let surfaceGeneration = 0;

// ══════════════════════════════════════════════════════════════
//  Initialisation
// ══════════════════════════════════════════════════════════════
window.addEventListener('DOMContentLoaded', function() {
  buildPocketCards();
  if (NOPOCKETS) { showNoPockets(); }
  updateResidueBtn();
  initViewer();
});

function initViewer() {
  viewer = $3Dmol.createViewer('viewer', {
    backgroundColor: '#06101e',
    antialias:       true
  });

  // ── Protein: fully opaque cartoon ─────────────────────────
  // opacity: 1.0 is critical — any value < 1 triggers 3Dmol's transparent
  // render path, which depth-sorts poorly alongside transparent surfaces
  // and makes the cartoon nearly invisible.
  proteinModel     = viewer.addModel(PROTEIN_PDB, 'pdb');
  PROTEIN_MODEL_ID = proteinModel.getID();
  viewer.setStyle(
    { model: PROTEIN_MODEL_ID },
    { cartoon: { colorscheme: 'chainHetatm', opacity: 1.0 } }
  );

  // ── Pocket models: per-pocket sphere PDBs ─────────────────
  POCKETS.forEach(function(p, i) {
    var m = viewer.addModel(p.pdb, 'pdb');
    viewer.setStyle(
      { model: m.getID() },
      { sphere: { color: p.color, opacity: 0.72, radius: sphereRadius } }
    );
    pocketModels.push(m);
    pocketVisible.push(true);
  });

  viewer.zoomTo();
  viewer.render();

  // Surface is computed on demand when the user moves the slider above 0.
  // We only fade the loading overlay here.
  setTimeout(fadeOutOverlay, 500);
}

// ── Re-apply protein styles after a surface operation ─────────────────────
function reapplyProteinStyles(delayMs) {
  setTimeout(function() {
    if (!viewer) return;
    viewer.setStyle(
      { model: proteinModel },
      { cartoon: { colorscheme: 'chainHetatm', opacity: 1.0 } }
    );
    if (showResidueSticks && selectedIdx !== null) {
      applyResidueSticks(selectedIdx, false);
    }
    viewer.render();
  }, delayMs !== undefined ? delayMs : 100);
}

// ── addSurface wrapper: handles both sync and Promise return values ────────
// Modern 3Dmol.js returns a Promise from addSurface(). Older versions return
// the surface ID directly. Passing a Promise to setSurfaceMaterialStyle()
// silently fails — that is why the opacity slider appeared to do nothing.
// This wrapper normalises both paths so surfaceObj is always a real ID.
function addSurfaceAsync(opacity, afterFn) {
  var result = viewer.addSurface(
    $3Dmol.SurfaceType.VDW,
    { opacity: opacity, color: '#4a90c8' },
    { model: PROTEIN_MODEL_ID }
  );
  if (result && typeof result.then === 'function') {
    result.then(function(id) { surfaceObj = id; if (afterFn) afterFn(); });
  } else {
    surfaceObj = result;
    if (afterFn) afterFn();
  }
}

function computeSurface() {
  // Claim this generation slot.  Any older in-flight callback will see
  // myGen !== surfaceGeneration and abort without touching the scene.
  surfaceGeneration++;
  var myGen = surfaceGeneration;

  // Show loading overlay while the mesh is generated
  var ov  = document.getElementById('loading-overlay');
  var txt = document.getElementById('loading-text');
  if (ov && txt) { txt.textContent = 'COMPUTING SURFACE'; ov.style.opacity = '1'; ov.style.display = 'flex'; }
  var dot = document.getElementById('surf-dot');
  var msg = document.getElementById('surf-msg');
  if (dot) dot.className = 'surf-dot';
  if (msg) msg.textContent = 'Computing\u2026';

  setTimeout(function() {
    // Abort if a newer computation or cancellation has arrived
    if (myGen !== surfaceGeneration) { fadeOutOverlay(); return; }

    try {
      if (surfaceObj !== null) { viewer.removeSurface(surfaceObj); surfaceObj = null; }

      addSurfaceAsync(surfOpacity, function() {
        // Check generation again inside the async callback — this is the
        // critical guard that prevents the race where the user drags back to
        // zero WHILE the surface mesh is still computing.
        if (myGen !== surfaceGeneration) {
          if (surfaceObj !== null) { viewer.removeSurface(surfaceObj); surfaceObj = null; }
          surfaceReady   = false;
          surfaceVisible = false;
          viewer.render();
          fadeOutOverlay();
          updateSurfaceBtn();
          return;
        }
        reapplyProteinStyles(0);
        surfaceReady   = true;
        surfaceVisible = true;
        fadeOutOverlay();
        var d = document.getElementById('surf-dot');
        var m = document.getElementById('surf-msg');
        if (d) d.className = 'surf-dot done';
        if (m) m.textContent = 'Surface active';
        updateSurfaceBtn();
      });

    } catch(e) {
      console.warn('[PocketFinder] Surface computation failed:', e);
      surfaceReady = false;
      fadeOutOverlay();
      var d = document.getElementById('surf-dot');
      var m = document.getElementById('surf-msg');
      if (d) d.className = 'surf-dot off';
      if (m) m.textContent = 'Surface unavailable';
      updateSurfaceBtn();
    }
  }, 60);
}

function fadeOutOverlay() {
  var ov = document.getElementById('loading-overlay');
  if (!ov) return;
  ov.style.opacity = '0';
  setTimeout(function() { ov.style.display = 'none'; }, 470);
}

// ══════════════════════════════════════════════════════════════
//  Pocket cards
// ══════════════════════════════════════════════════════════════
function buildPocketCards() {
  var list = document.getElementById('pocket-list');

  POCKETS.forEach(function(p, i) {
    var card = document.createElement('div');
    card.className = 'pocket-card';
    card.id = 'card-' + i;
    card.style.setProperty('--card-color', p.color);
    card.addEventListener('click', function() { selectPocket(i); });

    // Residues HTML (inline collapsible)
    var resHtml = '';
    if (p.residues && p.residues.length > 0) {
      var pills = p.residues.map(function(r) {
        return '<span class="res-pill"><span class="ch">' + r.chain + ':</span>' +
               r.resname + r.resnum + '</span>';
      }).join('');
      resHtml =
        '<details class="res-details" onclick="event.stopPropagation()">' +
          '<summary>' + p.residues.length + ' lining residue' +
            (p.residues.length !== 1 ? 's' : '') + '</summary>' +
          '<div class="res-grid">' + pills + '</div>' +
        '</details>';
    } else {
      resHtml = '<div class="res-none">No residue data available</div>';
    }

    card.innerHTML =
      '<div class="card-inner">' +
      '<div class="card-top">' +
        '<div class="card-name-row">' +
          '<span class="pocket-num">' + p.name + '</span>' +
          '<span class="pocket-id">' + p.id + '</span>' +
          (p.is_best ? '<span class="best-badge">BEST</span>' : '') +
          '<span class="deselect-hint" id="dh-' + i + '">click to deselect</span>' +
        '</div>' +
        '<div class="toggle-wrap on" id="toggle-' + i + '" ' +
               'onclick="onToggleClick(event,' + i + ')">' +
          '<div class="toggle-pill"></div>' +
          '<span class="toggle-lbl">show</span>' +
        '</div>' +
      '</div>' +

      '<div class="card-stats">' +
        '<div class="stat-row"><span class="stat-key">Volume</span>' +
          '<span class="stat-val c-blue">' + p.volume + ' \u00C5\u00B3</span></div>' +
        '<div class="stat-row"><span class="stat-key">Area</span>' +
          '<span class="stat-val c-green">' + p.area + ' \u00C5\u00B2</span></div>' +
        '<div class="stat-row"><span class="stat-key">Depth</span>' +
          '<span class="stat-val c-amber">' + p.depth + ' \u00C5</span></div>' +
        '<div class="stat-row"><span class="stat-key">Hydro.</span>' +
          '<span class="stat-val c-purple">' + p.hydrophobicity + '</span></div>' +
      '</div>' +

      '<div class="score-row">' +
        '<span class="score-key">SCORE</span>' +
        '<div class="score-track">' +
          '<div class="score-fill" id="sf-' + i + '"' +
               ' data-score="' + p.score + '"' +
               ' style="background:' + p.color + ';width:0%"></div>' +
        '</div>' +
        '<span class="score-num">' + p.score + '</span>' +
      '</div>' +

      resHtml +
      '</div>'; // .card-inner

    list.appendChild(card);
  });

  // Animate score bars after cards render
  setTimeout(function() {
    POCKETS.forEach(function(_, i) {
      var el = document.getElementById('sf-' + i);
      if (el) el.style.width = el.dataset.score + '%';
    });
  }, 400);
}

function showNoPockets() {
  var sec = document.getElementById('pocket-list');
  sec.innerHTML =
    '<div id="no-pockets">' +
      '<div class="np-icon">\u2B21</div>' +
      '<span>No binding pockets detected.</span>' +
      '<span style="font-size:11px;color:var(--text-dim);margin-top:4px;">' +
        'Try lowering <em>min_volume</em> and re-running.' +
      '</span>' +
    '</div>';
}

// ══════════════════════════════════════════════════════════════
//  Pocket selection / deselection
// ══════════════════════════════════════════════════════════════
function selectPocket(idx) {
  var wasSameCard = (selectedIdx === idx);

  // ── Visual: remove highlight from previously selected card ──
  if (selectedIdx !== null) {
    var prev = document.getElementById('card-' + selectedIdx);
    if (prev) prev.classList.remove('selected');
    // Clear residue sticks from the previously selected pocket
    if (showResidueSticks) { clearResidueSticks(selectedIdx); }
  }

  // ── Toggle off: clicking the same card deselects it ────────
  if (wasSameCard) {
    selectedIdx = null;
    showResidueSticks = false;
    // Remove the selected class (it was already removed above) and
    // fade the card back to its normal state cleanly.
    updateResidueBtn();
    // Zoom back to the full protein with a smooth animation
    if (viewer) { viewer.zoomTo({}, 900); viewer.render(); }
    return;
  }

  // ── Select new card ─────────────────────────────────────────
  selectedIdx = idx;
  var card = document.getElementById('card-' + idx);
  if (card) card.classList.add('selected');

  // Zoom to pocket centroid / model and do flash animation
  if (viewer && pocketModels[idx]) {
    viewer.zoomTo({ model: pocketModels[idx].getID() }, 900);
    if (pocketVisible[idx]) {
      var p = POCKETS[idx];
      viewer.setStyle(
        { model: pocketModels[idx].getID() },
        { sphere: { color: p.color, opacity: 1.0, radius: sphereRadius * 1.4 } }
      );
      viewer.render();
      setTimeout(function() {
        if (viewer && pocketVisible[idx]) {
          viewer.setStyle(
            { model: pocketModels[idx].getID() },
            { sphere: { color: p.color, opacity: 0.82, radius: sphereRadius } }
          );
          viewer.render();
        }
      }, 750);
    } else {
      viewer.render();
    }
  }

  // If residue sticks were active for a previous pocket, re-apply for the new one
  if (showResidueSticks) {
    applyResidueSticks(idx, true);
  }
  updateResidueBtn();
}

// ══════════════════════════════════════════════════════════════
//  Residue stick visualisation
// ══════════════════════════════════════════════════════════════

// Apply stick style to the lining residues of pocket at index idx.
// Pass doRender=true to call viewer.render() at the end.
//
// Design choices:
//   • viewer.addStyle()  — ADDS sticks on top of the existing cartoon without
//     touching it. No need to re-specify the cartoon style; no risk of removing
//     it accidentally.
//   • proteinModel object in selector (not integer ID) — the GLModel object is
//     always unambiguous; passing an integer ID can silently fail in some 3Dmol
//     versions when combined with chain + resi atom-level filters.
//   • Batch residues per chain into a single addStyle call with a resi array —
//     fewer calls, less chance of intermediate render stomping.
//   • Jmol colorscheme for sticks — standard CPK element colours (N=blue,
//     O=red, S=yellow, C=grey…) so residue chemistry is immediately readable.
function applyResidueSticks(idx, doRender) {
  if (!viewer || !proteinModel) return;
  var p = POCKETS[idx];
  if (!p.residues || p.residues.length === 0) return;

  // Group residue numbers by chain
  var chainMap = {};
  p.residues.forEach(function(r) {
    var chain = (r.chain || '').trim();
    var resn  = parseInt(r.resnum, 10);
    if (!chain || chain === '?' || isNaN(resn)) return;
    if (!chainMap[chain]) chainMap[chain] = [];
    chainMap[chain].push(resn);
  });

  if (Object.keys(chainMap).length === 0) return;

  // One addStyle call per chain — sticks layered on top of cartoon
  // colorscheme 'Jmol' gives standard CPK element colours:
  //   C = grey, N = blue, O = red, S = yellow, H = white, P = orange …
  Object.keys(chainMap).forEach(function(chain) {
    viewer.addStyle(
      { model: proteinModel, chain: chain, resi: chainMap[chain] },
      { stick: { colorscheme: 'Jmol', radius: 0.25, singleBond: false } }
    );
  });

  if (doRender) viewer.render();
}

// Remove stick style from all lining residues, restoring cartoon-only.
// Rather than trying to selectively un-style individual residues (fragile),
// we reset the entire protein model to cartoon-only in one call. This
// guarantees a clean state regardless of how addStyle layered the sticks.
function clearResidueSticks(idx) {
  if (!viewer || !proteinModel) return;
  viewer.setStyle(
    { model: proteinModel },
    { cartoon: { colorscheme: 'chainHetatm', opacity: 1.0 } }
  );
  viewer.render();
}

// Button handler: toggle residue sticks for the currently selected pocket
function toggleResidueSticks() {
  if (selectedIdx === null) {
    showToast('Select a pocket first to view its lining residues');
    return;
  }
  showResidueSticks = !showResidueSticks;
  if (showResidueSticks) {
    applyResidueSticks(selectedIdx, true);
  } else {
    clearResidueSticks(selectedIdx);
  }
  updateResidueBtn();
}

// Sync the residue-stick button visual state.
// Three states:
//   idle    – no pocket selected: dimmed, click shows a toast hint
//   ready   – pocket selected, sticks off: normal, gentle pulse to attract attention
//   active  – sticks are showing: accent colour, "active" class
function updateResidueBtn() {
  var hasSelection = (selectedIdx !== null);
  var active       = (showResidueSticks && hasSelection);

  // ── toolbar button ──────────────────────────────────────────────────────
  var btn = document.getElementById('res-stick-btn');
  var lbl = document.getElementById('res-stick-label');
  if (btn) {
    btn.disabled = false;  // always clickable so we can show a helpful hint
    if (!hasSelection) {
      btn.classList.remove('active', 'res-ready');
      btn.style.opacity = '0.42';
      if (lbl) lbl.textContent = 'Residues';
    } else if (!showResidueSticks) {
      btn.classList.remove('active');
      btn.classList.add('res-ready');
      btn.style.opacity = '1';
      if (lbl) lbl.textContent = 'Residues';
    } else {
      btn.classList.remove('res-ready');
      btn.classList.add('active');
      btn.style.opacity = '1';
      if (lbl) lbl.textContent = 'Hide Residues';
    }
  }

  // ── sidebar button ──────────────────────────────────────────────────────
  var sideBtn  = document.getElementById('res-stick-sidebar-btn');
  var hintEl   = document.getElementById('res-btn-hint');
  var iconEl   = document.getElementById('res-btn-icon');
  var textEl   = document.getElementById('res-btn-text');
  if (sideBtn) {
    sideBtn.classList.toggle('active',  active);
    sideBtn.classList.toggle('waiting', !hasSelection);
  }
  if (iconEl) iconEl.textContent = active ? '◆' : '⬡';
  if (textEl) textEl.textContent = active ? 'Hide pocket residues' : 'Show pocket residues in 3D';
  if (hintEl) {
    if (!hasSelection) {
      hintEl.textContent   = '← select a pocket card first';
      hintEl.style.opacity = '1';
    } else if (active) {
      hintEl.textContent   = 'residues shown as sticks in viewer';
      hintEl.style.opacity = '1';
    } else {
      hintEl.textContent   = '';
      hintEl.style.opacity = '0';
    }
  }

  // ── deselect hints inside cards ─────────────────────────────────────────
  // Show "click again to deselect" on the selected card only
  document.querySelectorAll('.deselect-hint').forEach(function(el) {
    el.style.opacity = '0';
  });
  if (hasSelection) {
    var dh = document.getElementById('dh-' + selectedIdx);
    if (dh) dh.style.opacity = '1';
  }
}

// ══════════════════════════════════════════════════════════════
//  Pocket toggle
// ══════════════════════════════════════════════════════════════
function onToggleClick(event, idx) {
  event.stopPropagation();
  var isNowOn = !pocketVisible[idx];
  pocketVisible[idx] = isNowOn;
  var label = document.getElementById('toggle-' + idx);
  if (label) label.classList.toggle('on', isNowOn);
  applyPocketStyle(idx, isNowOn);
}

function applyPocketStyle(idx, visible) {
  if (!viewer || !pocketModels[idx]) return;
  var p = POCKETS[idx];
  viewer.setStyle(
    { model: pocketModels[idx].getID() },
    visible ? { sphere: { color: p.color, opacity: 0.72, radius: sphereRadius } } : {}
  );
  viewer.render();
}

function toggleAllPockets(show) {
  POCKETS.forEach(function(_, i) {
    pocketVisible[i] = show;
    var lbl = document.getElementById('toggle-' + i);
    if (lbl) lbl.classList.toggle('on', show);
    applyPocketStyle(i, show);
  });
}

// ══════════════════════════════════════════════════════════════
//  Sphere radius control
// ══════════════════════════════════════════════════════════════
function onSphereRadius(val) {
  sphereRadius = parseFloat(val);
  document.getElementById('sphere-val').textContent = sphereRadius.toFixed(2);
  if (!viewer) return;
  POCKETS.forEach(function(p, i) {
    if (!pocketVisible[i]) return;
    viewer.setStyle(
      { model: pocketModels[i].getID() },
      { sphere: { color: p.color, opacity: 0.72, radius: sphereRadius } }
    );
  });
  viewer.render();
}

// ══════════════════════════════════════════════════════════════
//  Surface controls
// ══════════════════════════════════════════════════════════════
function onSurfaceOpacity(val) {
  surfOpacity = parseFloat(val);
  var valEl = document.getElementById('surf-val');
  if (valEl) valEl.textContent = surfOpacity === 0 ? 'off' : surfOpacity.toFixed(2);
  if (!viewer) return;

  if (surfOpacity === 0) {
    // ── Turn off ───────────────────────────────────────────────────────────
    // Increment generation FIRST so any in-flight computeSurface() callback
    // sees a stale generation number and aborts before adding a surface.
    surfaceGeneration++;
    if (surfaceObj !== null) { viewer.removeSurface(surfaceObj); surfaceObj = null; }
    surfaceReady   = false;
    surfaceVisible = false;
    viewer.render();

  } else if (surfaceObj === null) {
    // ── Surface not yet computed (first use or after removal) ──────────────
    computeSurface();

  } else {
    // ── Surface already exists: update opacity in-place ───────────────────
    // setSurfaceMaterialStyle mutates only the render material — the mesh is
    // NOT recomputed and the cartoon is never touched.
    viewer.setSurfaceMaterialStyle(surfaceObj, { opacity: surfOpacity, color: '#4a90c8' });
    surfaceVisible = true;
    viewer.render();
  }
  updateSurfaceBtn();
}

function onSurfaceToggle() {
  if (!viewer) return;
  surfaceVisible = !surfaceVisible;
  if (surfaceVisible) {
    var slider = document.getElementById('surf-slider');
    surfOpacity = slider ? (parseFloat(slider.value) || 0.15) : 0.15;
    if (surfOpacity === 0) { surfOpacity = 0.15; if (slider) slider.value = '0.15'; }
    var valEl = document.getElementById('surf-val');
    if (valEl) valEl.textContent = surfOpacity.toFixed(2);
    computeSurface();   // always recomputes when toggling on
  } else {
    if (surfaceObj !== null) { viewer.removeSurface(surfaceObj); surfaceObj = null; }
    surfaceGeneration++;   // cancel any in-flight computation
    surfaceReady = false;
    viewer.render();
  }
  updateSurfaceBtn();
}

function updateSurfaceBtn() {
  var btn = document.getElementById('surf-toggle-btn');
  if (!btn) return;
  var label = 'Surface: ' + (surfaceVisible && surfaceReady ? 'ON' : 'OFF');
  var svgStr =
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">' +
    '<circle cx="12" cy="12" r="10"/>' +
    '<path d="M12 2a14.5 14.5 0 0 0 0 20 14.5 14.5 0 0 0 0-20"/>' +
    '</svg>';
  btn.innerHTML = svgStr + label;
  if (surfaceVisible && surfaceReady) {
    btn.classList.add('active');
  } else {
    btn.classList.remove('active');
  }

  var dot = document.getElementById('surf-dot');
  var msg = document.getElementById('surf-msg');
  if (!surfaceReady) return;
  if (dot) dot.className = surfaceVisible ? 'surf-dot done' : 'surf-dot off';
  if (msg) msg.textContent = surfaceVisible ? 'Surface active' : 'Surface hidden';
}

// ══════════════════════════════════════════════════════════════
//  Reset view
// ══════════════════════════════════════════════════════════════
function resetView() {
  if (!viewer) return;
  // Clear residue sticks if active
  if (showResidueSticks && selectedIdx !== null) {
    clearResidueSticks(selectedIdx);
  }
  showResidueSticks = false;
  selectedIdx = null;
  document.querySelectorAll('.pocket-card').forEach(function(c) {
    c.classList.remove('selected');
  });
  updateResidueBtn();
  viewer.zoomTo();
  viewer.render();
}
// ══════════════════════════════════════════════════════════════
//  Toast helper
// ══════════════════════════════════════════════════════════════
var _toastTimer = null;
function showToast(msg, durationMs) {
  durationMs = durationMs || 2600;
  var el = document.getElementById('toast');
  if (!el) return;
  if (_toastTimer) { clearTimeout(_toastTimer); }
  el.textContent = msg;
  el.classList.add('show');
  _toastTimer = setTimeout(function() { el.classList.remove('show'); }, durationMs);
}

</script>
</body>
</html>"""


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("Pocket Finder — pyKVFinder Dashboard")
    print("=" * 50)

    parser = argparse.ArgumentParser(
        prog="pocket_finder",
        description="PyKVFinder-based tool to identify and visualise binding pockets in a PDB file",
        epilog="Villoutreix group",
    )
    parser.add_argument(
        "-p", "--protein",
        action="store",
        required=True,
        help="PDB file of the protein to analyse",
    )
    parser.add_argument(
        "-v", "--min-volume",
        type=float,
        default=100.0,
        metavar="VOL",
        help="Minimum pocket volume in Å³ (default: 100)",
    )
    args = parser.parse_args()

    protein   = args.protein
    home_dir  = os.getcwd()

    # ── run pyKVFinder ────────────────────────────────────────────────────────
    finder = PocketFinder(work_dir=home_dir)
    result = finder.detect_pockets(protein, min_volume=args.min_volume)

    if not result["success"]:
        print(f"[ERROR] {result.get('error', 'Unknown error')}")
        raise SystemExit(1)

    pockets     = result["pockets"]
    cavity_pdb  = result.get("cavity_pdb", "")

    # ── build 3-D dashboard ───────────────────────────────────────────────────
    protein_stem   = Path(protein).stem
    dashboard_file = f"pocket_dashboard_{protein_stem}.html"

    print(f"\n[DASHBOARD] Building interactive 3D dashboard …")
    dashboard_html = create_pocket_dashboard_html(
        pockets          = pockets,
        protein_pdb_path = protein,
        cavity_pdb_path  = cavity_pdb,
        protein_name     = protein_stem.upper(),
    )

    with open(dashboard_file, "w", encoding="utf-8") as fh:
        fh.write(dashboard_html)

    print(f"[DASHBOARD] Saved to: {dashboard_file}")
    print(f"[DASHBOARD] Open in your browser to explore {len(pockets)} pockets interactively.\n")

    # ── status ────────────────────────────────────────────────────────────────
    if PYKVFINDER_AVAILABLE:
        print("✓ pyKVFinder available")
    else:
        print("✗ pyKVFinder not installed")
        print("  Install with: pip install pyKVFinder")
