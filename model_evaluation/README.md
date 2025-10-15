# Model Evaluation Pipeline

This directory contains Jupyter notebooks for evaluating the performance of pocket predictors and docking results on the PoseBustes, Astex, Coach420, and BU48 datasets.

## Overview of Evaluation Datasets

The pocket predictors were evaluated based on docking accuracy and pocket-ligand intersection (PLI) scores across different datasets:

- **PoseBusters Dataset & Astex Diverse Set**:
  - Evaluated using **docking accuracy**, including:
    - **RMSD < 2 Å**
    - **Passing all PoseBusters validation tests**
    - **Average PLI (pocket-ligand intersection) score**
  
- **Coach420 & BU48 Datasets**:
  - Evaluated **only** based on **PLI scores**.

### Required External Data

Before using these notebooks, users should first download the corresponding results from Zenodo repositories:

- **[Docking Results for Astex and PoseBusters when Guided by P2Rank, for comparison](https://zenodo.org/records/17354899)**
- **[Docking Results for the PoseBusters Dataset when Guided by Different Pocket Predictors](https://zenodo.org/records/14926719)**
- **[Docking Results for the Astex Diverse Set when Guided by Different Pocket Predictors](https://zenodo.org/records/14932535)**
- **[Pocket Predictions for Coach420 Dataset, Evaluated Based on PLI](https://zenodo.org/records/14933126)**
- **[Pocket Predictions for BU48 Dataset](https://zenodo.org/records/14933058)**

## Notebooks for Evaluation

Each notebook corresponds to a specific evaluation task:

- **PoseBusters Dataset Evaluation**
  - [`Evaluation_of_guided_docking_PoseBusters.ipynb`](Evaluation_of_guided_docking_PoseBusters.ipynb)
  - [`PoseBusters_PLI_evaluation.ipynb`](PoseBusters_PLI_evaluation.ipynb)

- **Astex Diverse Set Evaluation**
  - [`Evaluation_of_guided_docking_Astex.ipynb`](Evaluation_of_guided_docking_Astex.ipynb)
  - [`Astex_PLI_evaluation.ipynb`](Astex_PLI_evaluation.ipynb)

- **Coach420 Dataset Evaluation**
  - [`Coach420_PLI_evaluation.ipynb`](Coach420_PLI_evaluation.ipynb)

- **BU48 Dataset Evaluation**
  - [`BU48_PLI_evaluation.ipynb`](BU48_PLI_evaluation.ipynb)

## Comparison with PUResNet V2

To compare RAPID-Net with **PUResNet V2**, we performed predictions of likely interacting residues. However, for **large proteins such as 8F4J**, PUResNet V2 website failed to produce any predictions.

To fix this, we **assembled PUResNet V2** in the following notebook:

- [`PUResNet_V2_MinkowskiEngine_Colab_Installation_Fixed.ipynb`](PUResNet_V2_MinkowskiEngine_Colab_Installation_Fixed.ipynb)

Corresponding PUResNet V2 prediction outputs for 8F4J protein can be found here:

- **[Prediction of PUResNet V2, to Compare with RAPID-Net’s Predictions](https://zenodo.org/records/15001676)**

## Proteins Used in Demos

Several proteins from the **PoseBusters dataset**, as well as **ABHD5 (predicted by AlphaFold)**, were used in our demonstrations of RAPID-Net. These can be accessed at:

- **[Zenodo Repository for Demo Proteins](https://zenodo.org/records/14969445)**

---

This pipeline provides a comprehensive evaluation framework for assessing pocket prediction and docking accuracy across multiple datasets and models.

