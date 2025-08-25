#!/usr/bin/env python3
"""
Script to copy normalized_differential_steering_effects.svg files from 
probe_generation_results/b_neurips_workshop_results/{run_name}/d3_ordered_in_context_liar_projections/
to proj_figs/{run_name}.svg
"""

import os
import shutil
from pathlib import Path
import glob

def copy_svg_files():
    # Define the source pattern
    source_pattern = "probe_generation_results/b_neurips_workshop_results/*/d3_ordered_in_context_liar_projections/normalized_differential_steering_effects_simple.svg"
    
    # Find all matching files
    source_files = glob.glob(source_pattern)
    
    if not source_files:
        print("No files found matching the pattern.")
        print(f"Searched for: {source_pattern}")
        return
    
    print(f"Found {len(source_files)} files to copy:")
    
    for source_file in source_files:
        # Convert to Path object for easier manipulation
        source_path = Path(source_file)
        
        # Extract the run_name (the directory name between b_neurips_workshop_results and d3_ordered...)
        parts = source_path.parts
        neurips_index = parts.index('b_neurips_workshop_results')
        run_name = parts[neurips_index + 1]
        
        # Create destination path
        dest_dir = Path('proj_figs')
        dest_name = f'{run_name}.svg'
        dest_file = dest_dir / dest_name
        
        # Create destination directory if it doesn't exist
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy the file
        shutil.copy2(source_file, dest_file)

        print(
"""
\\begin{figure}
    \centering
    \includesvg[width=\linewidth]{figures/neural_projections/""" +  dest_name + """}
    \caption{""" + dest_name + """}
    \label{fig:+""" +  dest_name + """}
\end{figure}
"""
        )

if __name__ == "__main__":
    copy_svg_files()