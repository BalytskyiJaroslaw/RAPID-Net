# RAPID-Net: Accurate Pocket Identification for Binding-Site-Agnostic Docking  

![RAPID-Net](https://raw.githubusercontent.com/BalytskyiJaroslaw/RAPID-Net/main/RAPID_combine.png)

In this repository, we implement **RAPID-Net**, a Deep Learning model for accurate protein-ligand pocket identification, leading to improved ligand docking in the absence of prior knowledge of the binding site location. This work corresponds to our paper:

**[Accurate Pocket Identification for Binding-Site-Agnostic Docking](https://arxiv.org/abs/2502.02371)**  
*Yaroslav Balytskyi, Inna Hubenko, Alina Balytska, Christopher V. Kelly*  
arXiv preprint, 2025.  

## Model Overview  

Unlike other Deep Learning models, RAPID-Net uses **soft labels** to account for uncertainty in pocket boundaries and **L² Soft Dice loss function**, which provides more accurate and generalized pocket predictions. Additionally, our model integrates an attention mechanism. Interestingly, we found that removing redundant residual connections enhances performance, thus we refined the architecture accordingly. The model's design is illustrated below:

![RAPID-Net Architecture](https://github.com/BalytskyiJaroslaw/RAPID-Net/blob/main/RAPID_diagram_insert.png)  

These predicted pockets provide accurate initial approximations for AutoDock Vina search grids, greatly improving docking efficiency and accuracy, as illustrated by the following example: 

## Pocket Prediction & Guided Docking Example by AutoDock Vina for 8F4J protein which AlphaFold 3 cannot process as a whole (but is processed smoothly by RAPID-Net 😊)

A complete example of pocket prediction for 8F4J and its use for guided docking is provided in this **Jupyter Notebook**:  

[🔗 Zenodo Link: Pocket Prediction & Guided Docking for Protein 8F4J](https://zenodo.org/records/15026755)  

8F4J protein is illustrated below:
!["Blind" docking for 8F4J protein](https://github.com/BalytskyiJaroslaw/RAPID-Net/blob/main/8F4J_PHO_combined.png) 

In a similar way, using the same notebook, one can perform "blind" docking for any protein in PoseBusters and Astex Diverse Set.

A complete example of pocket prediction for ABHD5 protein, generation of initial ligand conformation from IUPAC or SMILES, the guided docking, and PLIP interaction profiling is provided in this **Jupyter Notebook**:  

!["Blind" docking for ABHD5 protein](https://github.com/BalytskyiJaroslaw/RAPID-Net/blob/main/Vina_Setup.png) 

## Citation  

If you use RAPID-Net in your research, please cite our paper:  

```bibtex
@article{balytskyi2025accurate,
  title={Accurate Pocket Identification for Binding-Site-Agnostic Docking},
  author={Balytskyi, Yaroslav and Hubenko, Inna and Balytska, Alina and Kelly, Christopher V},
  journal={arXiv preprint arXiv:2502.02371},
  year={2025}
}

## Thanks

We use the following software tools for processing proteins and ligands, as well as for their visualization:

## Citations

This repository makes use of several software tools and datasets. Please cite the following when using this repository:

- **OpenBabel**  
  O'Boyle, N. M., Banck, M., James, C. A., Morley, C., Vandermeersch, T., Hutchison, G. R. (2011). Open Babel: An open chemical toolbox. *Journal of Cheminformatics, 3*, 33. [https://doi.org/10.1186/1758-2946-3-33](https://doi.org/10.1186/1758-2946-3-33)

- **RDKit**  
  RDKit: Open-source cheminformatics. [https://www.rdkit.org](https://www.rdkit.org)

- **Reduce**  
  Word, J. M., Lovell, S. C., Richardson, J. S., & Richardson, D. C. (1999). Asparagine and glutamine: Using hydrogen atom contacts in the choice of side-chain amide orientation. *Journal of Molecular Biology, 285*(4), 1735-1747. [https://doi.org/10.1006/jmbi.1998.2401](https://doi.org/10.1006/jmbi.1998.2401)

- **Meeko**  
  For ligand preparation with AutoDock-GPU and AutoDock Vina. GitHub repository: [https://github.com/forlilab/Meeko](https://github.com/forlilab/Meeko)

- **PoseBusters**  
  Durrant, J. D., & McCammon, J. A. (2022). PoseBusters: An automatic tool for evaluating the accuracy of protein-ligand complexes in molecular docking. *Journal of Chemical Information and Modeling, 62*(12), 2934–2943. [https://doi.org/10.1021/acs.jcim.2c00387](https://doi.org/10.1021/acs.jcim.2c00387)

- **LaboDock**  
  GitHub repository: [https://github.com/milobioinformatics/LaboDock](https://github.com/milobioinformatics/LaboDock)

- **py3Dmol**  
  py3Dmol: An interactive 3D molecular visualization tool for Jupyter Notebooks. GitHub repository: [https://github.com/3dmol/3Dmol.js](https://github.com/3dmol/3Dmol.js)

- **PyMOL**  
  Schrödinger, LLC. (2015). The PyMOL Molecular Graphics System, Version 2.0. [https://pymol.org](https://pymol.org)

