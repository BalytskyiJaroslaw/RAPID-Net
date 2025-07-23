# RAPID-Net: Accurate Pocket Identification for Binding-Site-Agnostic Docking  


<a href="https://github.com/BalytskyiJaroslaw/RAPID-Net/blob/main/New_Logo.png" target="_blank" style="float: right; margin-left: 15px;">
  <img src="https://raw.githubusercontent.com/BalytskyiJaroslaw/RAPID-Net/main/New_Logo.png" alt="RAPID-Net" width="120"/>
</a>

In this repository, we implement **RAPID-Net**, a Deep Learning model for accurate protein-ligand pocket identification, leading to improved ligand docking in the absence of prior knowledge of the binding site location. Additionally, RAPID-Net can identify remote protein sites that indirectly affect ligand binding, providing a deeper understanding of protein-ligand interactions.

This work corresponds to our paper:

➡ **[Accurate Pocket Identification for Binding-Site-Agnostic Docking](https://arxiv.org/abs/2502.02371)**  
*Yaroslav Balytskyi, Inna Hubenko, Alina Balytska, Christopher V. Kelly*  
arXiv preprint, 2025.  

## Model Overview  

Unlike other Deep Learning models, RAPID-Net uses **soft labels** to account for uncertainty in pocket boundaries and **L² Soft Dice loss function**, which provides more accurate and generalized pocket predictions. Additionally, our model integrates an attention mechanism. Interestingly, we found that removing redundant residual connections enhances performance, thus we refined the architecture accordingly. The model's design is illustrated below:

![RAPID-Net Architecture](https://github.com/BalytskyiJaroslaw/RAPID-Net/blob/main/RAPID_diagram_insert.png)  

These predicted pockets provide accurate initial approximations for AutoDock Vina search grids, greatly improving docking efficiency and accuracy, as illustrated by the following example: 

## Pocket Prediction & Guided Docking Example for 8F4J protein which AlphaFold 3 cannot process as a whole, but RAPID-Net handles it smoothly! 😊

A complete example of pocket prediction for 8F4J and its use for guided docking is provided in this **Jupyter Notebook**:  

➡ **[8F4J Protein Docking Demonstration (Zenodo)](https://zenodo.org/records/15026755)**

8F4J protein is illustrated below:
!["Blind" docking for 8F4J protein](https://github.com/BalytskyiJaroslaw/RAPID-Net/blob/main/8F4J.png) 

In a similar way, using the same notebook, one can perform "blind" docking for any protein in PoseBusters and Astex Diverse Set.

## ABHD5 Docking Demonstration

A complete example of pocket prediction for ABHD5 protein, generation of initial ligand conformation from IUPAC or SMILES, the guided docking, and PLIP interaction profiling is provided in this **Jupyter Notebook**: 

➡ **[ABHD5 Protein Docking Demonstration (Zenodo)](https://zenodo.org/records/15048009)**

A **clean version** of the notebook is available, which works for **any protein** (without the ABHD5-specific output).  

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BalytskyiJaroslaw/RAPID-Net/blob/main/Demo_docking_guided_by_RAPID_Net_ABHD5_as_an_example_submit_clean.ipynb)

An example of grid generation based on the predicted pocket is shown below:
!["Blind" docking for ABHD5 protein](https://github.com/BalytskyiJaroslaw/RAPID-Net/blob/main/Vina_Setup.png) 

## Pocket prediction and its usage for guided docking, for 8F4J protein from PoseBusters

## Watch the Video of docking in real time here:
[![First Slide](https://github.com/BalytskyiJaroslaw/RAPID-Net/blob/main/8F4J_screenshot_1.png?raw=true)](https://youtu.be/EkUKmoW11pE)


## Citation  

If you use RAPID-Net in your research, please cite our paper:  

@article{balytskyi2025accurate,
  title = {Accurate Pocket Identification for Binding-Site-Agnostic Docking},
  author = {Balytskyi, Yaroslav and Hubenko, Inna and Balytska, Alina and Kelly, Christopher V.},
  journal = {arXiv preprint arXiv:2502.02371},
  year = {2025}
}

## Thanks

We use the following software tools for processing proteins and ligands, as well as for their visualization:

- **OpenBabel**  
  O'Boyle, N. M., Banck, M., James, C. A., Morley, C., Vandermeersch, T., Hutchison, G. R. (2011). Open Babel: An open chemical toolbox. *Journal of Cheminformatics, 3*, 33. [https://doi.org/10.1186/1758-2946-3-33](https://doi.org/10.1186/1758-2946-3-33)

- **RDKit**  
  RDKit: Open-source cheminformatics. [https://www.rdkit.org](https://www.rdkit.org)

- **Reduce**  
  *Reduce* is a tool for adding and correcting hydrogen atoms in PDB files.  
  We used an older version of Reduce:  
  [https://github.com/rlabduke/reduce](https://github.com/rlabduke/reduce).  

  A newer version, **Reduce2**, is available here:  
  [https://github.com/cctbx/cctbx_project/tree/master/mmtbx/reduce](https://github.com/cctbx/cctbx_project/tree/master/mmtbx/reduce).

- **Meeko**  
 Interface for AutoDock, molecule parameterization. GitHub repository: [https://github.com/forlilab/Meeko](https://github.com/forlilab/Meeko)

- **PoseBusters**  
  Buttenschoen, M., Morris, G. M., & Deane, C. M. (2024).  
  *PoseBusters: AI-Based Docking Methods Fail to Generate Physically Valid Poses or Generalise to Novel Sequences.*  
  Chemical Science, 15(9), 3130–3139.  
  [https://doi.org/10.1039/D3SC06192H](https://doi.org/10.1039/D3SC06192H)

- **PLIP (Protein-Ligand Interaction Profiler)**  
  PLIP: A fully automated tool for analyzing protein-ligand interactions.  
  GitHub Repository: [https://github.com/pharmai/plip](https://github.com/pharmai/plip)  
  Official Website: [https://plip-tool.biotec.tu-dresden.de](https://plip-tool.biotec.tu-dresden.de)  
  Reference: Salentin et al., 2015, "PLIP: Fully Automated Protein–Ligand Interaction Profiler," *Nucleic Acids Research*, 43(W1): W443–W447. [DOI: 10.1093/nar/gkv315](https://doi.org/10.1093/nar/gkv315)

- **LaboDock**  
  LABODOCK: A Colab-Based Molecular Docking Tools: [[https://github.com/milobioinformatics/LaboDock](https://github.com/RyanZR/labodock)]([https://github.com/milobioinformatics/LaboDock](https://github.com/RyanZR/labodock))

- **py3Dmol**  
  py3Dmol: An interactive 3D molecular visualization tool for Jupyter Notebooks. GitHub repository: [https://github.com/3dmol/3Dmol.js](https://github.com/3dmol/3Dmol.js)

- **PyMOL**  
  Schrödinger, LLC. (2015). The PyMOL Molecular Graphics System, Version 2.0. [https://pymol.org](https://pymol.org)

## Contact

For any **questions, concerns, or bug reports**, feel free to reach out:  

📧 **hr6998@wayne.edu**  
📧 **ybalytsk@uccs.edu**  

---

😊🦁 **Happy pocket hunting—catch them like a lion!** 🎯🔬


