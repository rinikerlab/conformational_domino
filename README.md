# Conformational Domino: Linker modification enables control of key functional group orientation in macrocycles
This repository containts simulation data, input files and analysis notebook for the molecular dynamics simulations performed for this study.

## Publication
- Link to be added

## Abstract
Conformational preorganization is generally accepted as key aspect of macrocyclization, but how even minor modifications to a macrocyclic scaffold influence the conformational preorganization remains poorly understood. Here we show how macrocyclization and further derivatization of the linker region can improve affinity, selectivity, and plasma stability in a highly atom-efficient manner. A single, solvent exposed methyl group was found to improve binding affinity up to 10x over the non-methylated analog. This led to highly ligand-efficient macrocycles with a promising in vivo profile for the FK506-binding protein 51 (FKBP51), a key regulator of the human stress response. Using high-resolution co-crystal structures and molecular dynamics simulations, we found that small linker variations can be tuned to shift the orientation of a key carbonyl group into an advantageous position. This effect is specific to macrocycles, highlighting their potential for fine-tuned adjust-ments to enable desired properties.

## Usage
```
# Install environment.
conda env create -f environment.yml
conda activate fkbp_mc_analysis

# Open /code/conformational_analysis.ipynb in your preferred Jupyter interface
```

## Repository structure
- `data` contains the starting points for the computational analysis, i.e. crystal structures of compounds **15a**, **29a** and **29b**, and the binding affinities of all compoounds.
- `code` contains:
    - GROMACS input files (topology + starting coordinates) both in water and in complex with FKBP51 for all systems, as well as MD parameters files `.mdp` in `md_input_files`
    - The `conformational_analysis.ipynb` notebook, a workflow that reproduces all the computational figures in the paper.
    - Python analysis functions in `mc_analysis.py` required by the notebook
    - Data extracted from the MD trajectories in `md_data`