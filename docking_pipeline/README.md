# Docking Pipeline

This directory contains Jupyter notebooks for preparing protein and ligand files and performing docking when prior knowledge of the binding site is available, as well as using various pocket predictors. This is an example for the PoseBusters dataset. For the Astex Diverse Set, all steps are completely analogous to this one.

## Notebooks Overview

### 1. `Prepare_Proteins_and_Ligands_for_docking.ipynb` 
This notebook is dedicated to preparing the protein and ligand files for docking:

- **Protein processing:**
  - Protonation of the protein structure using Reduce.
  - Removal of bad atoms.
  - Generation of the `.pdbqt` file for docking using OpenBabel.
  - LaboDock was used for visualization of proteins, their pockets, and search grids.

- **Ligand processing:**
  - Addition of hydrogen atoms using rdkit.
  - Generation of the `.pdbqt` file for docking by Meeko. 

### 2. `Docking_when_prior_knowledge_available.ipynb`
This notebook performs docking using **AutoDock Vina** when prior knowledge is available. In this case, docking is guided by a reference ligand used to center the search box as:

- The docking box is set up with dimensions **25 Å × 25 Å × 25 Å** centered on the reference ligand.

### 3. `Docking_when_guided_by_single_Model.ipynb`
This notebook is dedicated to docking the **PoseBusters dataset** when using a **single neural network** for pocket prediction. The following models are tested:

- **Kalasanty**
- **PUResNet V1**
- **RAPID-Net**

**Thresholds used for pocket prediction:** `[2, 5, 10, 15]`

### 4. `Guided_Docking_by_ensembled_RAPID_Net.ipynb`
This notebook performs docking using an **ensembled version of 5 RAPID-Net models**, which improves overall docking performance.

- **Pocket classification thresholds:**
  - **For majority-voted pockets:** `[2, 5]`
  - **For minority-reported pockets:** `[2, 5, 10, 15]`

- **Pocket definitions:**
  - **Majority-voted pockets:** Composed of `2 Å × 2 Å × 2 Å` voxels that were predicted by at least **3 out of 5 RAPID-Net models**.
  - **Minimally-reported pockets:** Composed of voxels predicted by at least **1 out of 5 RAPID-Net models**.
 
For simplicity, when guided by a single model, the pockets were called **`Minimal`**, since there was only one model available. 

- **Consistency in classification:**
  - All models (**Kalasanty, PUResNet V1, RAPID-Net**) used the **same classification threshold of 0.5**. But the activation functions were different, sigmoid was used for Kalasanty and PUResNet while we used ReLU for RAPID-Net. See more details in: https://arxiv.org/abs/2502.02371

---

These notebooks provide a structured workflow for docking both when prior knowledge is available and when deep learning-based pocket prediction methods are used. When evaluating pocket predictors, reference ligand information is intentionally ignored. 

