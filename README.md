# RAPID-Net: Accurate Pocket Identification for Binding-Site-Agnostic Docking  

In this repository, we implement **RAPID-Net**, a Deep Learning model for accurate protein-ligand pocket identification, leading to improved ligand docking in the absence of prior knowledge of the binding site location. This work corresponds to our paper:

**[Accurate Pocket Identification for Binding-Site-Agnostic Docking](https://arxiv.org/abs/2502.02371)**  
*Yaroslav Balytskyi, Inna Hubenko, Alina Balytska, Christopher V. Kelly*  
arXiv preprint, 2025.  

## Model Overview  

Unlike other Deep Learning models, RAPID-Net uses **soft labels** to account for uncertainty in pocket boundaries and **L² Soft Dice loss function**, which provides more accurate and generalized pocket predictions. Additionally, our model integrates an attention mechanism. Interestingly, we found that removing redundant residual connections enhances performance, thus we refined the architecture accordingly. The model's design is illustrated below:

![RAPID-Net Architecture](https://github.com/BalytskyiJaroslaw/RAPID-Net/blob/main/RAPID_diagram_insert.pdf)  

These predicted pockets provide accurate initial approximations for AutoDock Vina search grids, greatly improving docking efficiency and accuracy, as illustrated by the following example: 

## Pocket Prediction & Guided Docking Example by AutoDock Vina for 8F4J protein which AlphaFold 3 cannot process as a whole (but is processed smoothly by RAPID-Net 😊)

A complete example of pocket prediction for 8F4J and its use for guided docking is provided in this **Jupyter Notebook**:  

[🔗 Zenodo Link: Pocket Prediction & Guided Docking for Protein 8F4J](https://zenodo.org/records/15026755)  

## Citation  

If you use RAPID-Net in your research, please cite our paper:  

```bibtex
@article{balytskyi2025accurate,
  title={Accurate Pocket Identification for Binding-Site-Agnostic Docking},
  author={Balytskyi, Yaroslav and Hubenko, Inna and Balytska, Alina and Kelly, Christopher V},
  journal={arXiv preprint arXiv:2502.02371},
  year={2025}
}
