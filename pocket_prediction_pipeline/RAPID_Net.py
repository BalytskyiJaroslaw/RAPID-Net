import tensorflow as tf
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, concatenate, BatchNormalization, Activation, add, SpatialDropout3D, GlobalAveragePooling3D, Reshape, Dense, multiply, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
import openbabel
from openbabel import pybel

import os
from Bio.PDB import PDBParser
import numpy as np


def identity_block(input_tensor, filters, stage, block):
    filters1, filters2, filters3 = filters
    bn_axis = -1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv3D(filters1, (1, 1, 1), kernel_initializer='he_normal', name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv3D(filters2, (3, 3, 3), padding='same', kernel_initializer='he_normal', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv3D(filters3, (1, 1, 1), kernel_initializer='he_normal', name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = add([x, input_tensor])
    x = Activation('relu')(x)
    return x

def conv_block(input_tensor, filters, stage, block, strides=(2, 2, 2)):
    filters1, filters2, filters3 = filters
    bn_axis = -1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv3D(filters1, (1, 1, 1), strides=strides, kernel_initializer='he_normal', name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv3D(filters2, (3, 3, 3), padding='same', kernel_initializer='he_normal', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv3D(filters3, (1, 1, 1), kernel_initializer='he_normal', name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv3D(filters3, (1, 1, 1), strides=strides, kernel_initializer='he_normal', name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = add([x, shortcut])
    x = Activation('relu')(x)
    return x

def se_block(input_tensor, ratio=16):
    channel_axis = -1
    filters = input_tensor.shape[channel_axis]

    se = GlobalAveragePooling3D()(input_tensor)
    se = Reshape((1, 1, 1, filters))(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    x = multiply([input_tensor, se])
    return x

def RAPID_Net(input_shape=(36, 36, 36, 18), filters=18, dropout_rate=0.5, l2_lambda=1e-3):
    params = {'kernel_size': 3, 'activation': 'relu', 'padding': 'same', 'kernel_regularizer': l2(l2_lambda)}

    inputs = Input(shape=input_shape, name='input')

    x = conv_block(inputs, [filters, filters, filters], stage=2, block='a', strides=(1, 1, 1))
    x1 = identity_block(x, [filters, filters, filters], stage=2, block='b')

    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(x1)

    x = conv_block(pool1, [filters*2, filters*2, filters*2], stage=4, block='a', strides=(1, 1, 1))
    x2 = identity_block(x, [filters*2, filters*2, filters*2], stage=4, block='b')

    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(x2)

    x = conv_block(pool2, [filters*4, filters*4, filters*4], stage=5, block='a', strides=(1, 1, 1))
    x3 = identity_block(x, [filters*4, filters*4, filters*4], stage=5, block='b')

    pool3 = MaxPooling3D(pool_size=(3, 3, 3))(x3)

    x = conv_block(pool3, [filters*8, filters*8, filters*8], stage=6, block='a', strides=(1, 1, 1))
    x4 = identity_block(x, [filters*8, filters*8, filters*8], stage=6, block='b')

    pool4 = MaxPooling3D(pool_size=(3, 3, 3))(x4)

    x = conv_block(pool4, [filters*16, filters*16, filters*16], stage=7, block='a', strides=(1, 1, 1))
    x = identity_block(x, [filters*16, filters*16, filters*16], stage=7, block='b')

    x = se_block(x)

    up6 = concatenate([UpSampling3D(size=(3, 3, 3))(x), x4], axis=-1)
    conv6 = Conv3D(filters=filters*8, **params)(up6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)

    conv6 = Conv3D(filters=filters*8, **params)(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)

    up7 = concatenate([UpSampling3D(size=(3, 3, 3))(conv6), x3], axis=-1)
    conv7 = Conv3D(filters=filters*4, **params)(up7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)

    conv7 = Conv3D(filters=filters*4, **params)(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)

    up8 = concatenate([UpSampling3D(size=(2, 2, 2))(conv7), x2], axis=-1)
    conv8 = Conv3D(filters=filters*2, **params)(up8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation('relu')(conv8)

    conv8 = Conv3D(filters=filters*2, **params)(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation('relu')(conv8)

    up9 = concatenate([UpSampling3D(size=(2, 2, 2))(conv8), x1], axis=-1)
    conv9 = Conv3D(filters=filters, **params)(up9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation('relu')(conv9)

    conv9 = Conv3D(filters=filters, **params)(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation('relu')(conv9)

    outputs = Conv3D(filters=1, kernel_size=1, kernel_regularizer=l2(1e-4), activation='relu', name='pocket')(conv9)

    model = Model(inputs=inputs, outputs=outputs, name='RAPID_Net')
    return model


def pocket_density_from_mol_RAPID_Net_run1(mol):
    if not isinstance(mol, pybel.Molecule):
        raise TypeError('mol should be a pybel.Molecule object, got %s '
                        'instead' % type(mol))
    if featurizer is None:
        raise ValueError('featurizer must be set to make predistions for '
                         'molecules')
    if scale is None:
        raise ValueError('scale must be set to make predistions')
    prot_coords, prot_features = featurizer.get_features(mol)
    centroid = prot_coords.mean(axis=0)
    prot_coords -= centroid


    resolution = 1. / scale
    x = make_grid(prot_coords, prot_features,
                              max_dist= max_dist,
                              grid_resolution=resolution)
    density = RAPID_Net_run1.predict(x)

    origin = (centroid - max_dist)
    step = np.array([1.0 / scale] * 3)

    return density, origin, step

def pocket_density_from_mol_RAPID_Net_run2(mol):
    if not isinstance(mol, pybel.Molecule):
        raise TypeError('mol should be a pybel.Molecule object, got %s '
                        'instead' % type(mol))
    if featurizer is None:
        raise ValueError('featurizer must be set to make predistions for '
                         'molecules')
    if scale is None:
        raise ValueError('scale must be set to make predistions')
    prot_coords, prot_features = featurizer.get_features(mol)
    centroid = prot_coords.mean(axis=0)
    prot_coords -= centroid


    resolution = 1. / scale
    x = make_grid(prot_coords, prot_features,
                              max_dist= max_dist,
                              grid_resolution=resolution)
    density = RAPID_Net_run2.predict(x)

    origin = (centroid - max_dist)
    step = np.array([1.0 / scale] * 3)

    return density, origin, step

def pocket_density_from_mol_RAPID_Net_run3(mol):
    if not isinstance(mol, pybel.Molecule):
        raise TypeError('mol should be a pybel.Molecule object, got %s '
                        'instead' % type(mol))
    if featurizer is None:
        raise ValueError('featurizer must be set to make predistions for '
                         'molecules')
    if scale is None:
        raise ValueError('scale must be set to make predistions')
    prot_coords, prot_features = featurizer.get_features(mol)
    centroid = prot_coords.mean(axis=0)
    prot_coords -= centroid


    resolution = 1. / scale
    x = make_grid(prot_coords, prot_features,
                              max_dist= max_dist,
                              grid_resolution=resolution)
    density = RAPID_Net_run3.predict(x)

    origin = (centroid - max_dist)
    step = np.array([1.0 / scale] * 3)

    return density, origin, step

def pocket_density_from_mol_RAPID_Net_run4(mol):
    if not isinstance(mol, pybel.Molecule):
        raise TypeError('mol should be a pybel.Molecule object, got %s '
                        'instead' % type(mol))
    if featurizer is None:
        raise ValueError('featurizer must be set to make predistions for '
                         'molecules')
    if scale is None:
        raise ValueError('scale must be set to make predistions')
    prot_coords, prot_features = featurizer.get_features(mol)
    centroid = prot_coords.mean(axis=0)
    prot_coords -= centroid


    resolution = 1. / scale
    x = make_grid(prot_coords, prot_features,
                              max_dist= max_dist,
                              grid_resolution=resolution)
    density = RAPID_Net_run4.predict(x)

    origin = (centroid - max_dist)
    step = np.array([1.0 / scale] * 3)

    return density, origin, step

def pocket_density_from_mol_RAPID_Net_run5(mol):
    if not isinstance(mol, pybel.Molecule):
        raise TypeError('mol should be a pybel.Molecule object, got %s '
                        'instead' % type(mol))
    if featurizer is None:
        raise ValueError('featurizer must be set to make predistions for '
                         'molecules')
    if scale is None:
        raise ValueError('scale must be set to make predistions')
    prot_coords, prot_features = featurizer.get_features(mol)
    centroid = prot_coords.mean(axis=0)
    prot_coords -= centroid


    resolution = 1. / scale
    x = make_grid(prot_coords, prot_features,
                              max_dist= max_dist,
                              grid_resolution=resolution)
    density = RAPID_Net_run5.predict(x)

    origin = (centroid - max_dist)
    step = np.array([1.0 / scale] * 3)

    return density, origin, step

from skimage.morphology import closing, label
from skimage.segmentation import clear_border
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt


def minimal_pockets_segmentation(density1, density2, density3, density4, density5, threshold=0.5, min_size=10, scale=0.5):

    voxel_size = (1 / scale) ** 3

    bw1 = closing((density1[0] > threshold).any(axis=-1))
    bw2 = closing((density2[0] > threshold).any(axis=-1))
    bw3 = closing((density3[0] > threshold).any(axis=-1))
    bw4 = closing((density4[0] > threshold).any(axis=-1))
    bw5 = closing((density5[0] > threshold).any(axis=-1))


    # Minimally-reported pockets, voted by at least one model
    combined_bw = np.sum([bw1, bw2, bw3, bw4, bw5], axis=0) >= 1

    # Apply morphological closing to reduce fragmentation
    combined_bw = ndi.binary_closing(combined_bw, structure=np.ones((3, 3, 3)))

    # Clear boundary-connected regions
    cleared = clear_border(combined_bw)

    # Label connected regions
    label_image, num_labels = label(cleared, return_num=True)

    for i in range(1, num_labels + 1):
        pocket_idx = (label_image == i)
        pocket_size = pocket_idx.sum() * voxel_size
        if pocket_size < min_size:
            label_image[np.where(pocket_idx)] = 0

    return label_image

def ensembled_pockets_segmentation(density1, density2, density3, density4, density5, threshold=0.5, min_size=50, scale=0.5):

    voxel_size = (1 / scale) ** 3

    bw1 = closing((density1[0] > threshold).any(axis=-1))
    bw2 = closing((density2[0] > threshold).any(axis=-1))
    bw3 = closing((density3[0] > threshold).any(axis=-1))
    bw4 = closing((density4[0] > threshold).any(axis=-1))
    bw5 = closing((density5[0] > threshold).any(axis=-1))

    # Majority-voted pockets, predicted by at least 3 models.
    combined_bw = np.sum([bw1, bw2, bw3, bw4, bw5], axis=0) >= 3

    # Apply morphological closing to reduce fragmentation
    combined_bw = ndi.binary_closing(combined_bw, structure=np.ones((3, 3, 3)))

    cleared = clear_border(combined_bw)

    label_image, num_labels = label(cleared, return_num=True)

    for i in range(1, num_labels + 1):
        pocket_idx = (label_image == i)
        pocket_size = pocket_idx.sum() * voxel_size
        if pocket_size < min_size:
            label_image[np.where(pocket_idx)] = 0

    return label_image


def save_pocket_mol2_RAPID_Net_Majority(mol, path, format, **pocket_kwargs):

    density1, origin1, step1 = pocket_density_from_mol_RAPID_Net_run1(mol)
    density1 = np.clip(density1, 0, 1)

    density2, origin2, step2 = pocket_density_from_mol_RAPID_Net_run2(mol)
    density2 = np.clip(density2, 0, 1)

    density3, origin3, step3 = pocket_density_from_mol_RAPID_Net_run3(mol)
    density3 = np.clip(density3, 0, 1)

    density4, origin4, step4 = pocket_density_from_mol_RAPID_Net_run4(mol)
    density4 = np.clip(density4, 0, 1)

    density5, origin5, step5 = pocket_density_from_mol_RAPID_Net_run5(mol)
    density5 = np.clip(density5, 0, 1)

    pockets = ensembled_pockets_segmentation(density1, density2, density3, density4, density5, threshold=0.5, min_size=50, scale=0.5)

    i = 0
    for pocket_label in range(1, pockets.max() + 1):
        indices = np.argwhere(pockets == pocket_label).astype('float')
        indices *= step1
        indices += origin1
        mol = openbabel.OBMol()
        for idx in indices:
            a = mol.NewAtom()
            a.SetVector(float(idx[0]), float(idx[1]), float(idx[2]))
        p_mol = pybel.Molecule(mol)
        # Need to modify this line
        p_mol.write(format, path + '/pocket_thr05_Majority' + str(i) + '.' + format)
        i += 1

def save_pocket_mol2_RAPID_Net_Minimal(mol, path, format, **pocket_kwargs):

    density1, origin1, step1 = pocket_density_from_mol_RAPID_Net_run1(mol)
    density1 = np.clip(density1, 0, 1)

    density2, origin2, step2 = pocket_density_from_mol_RAPID_Net_run2(mol)
    density2 = np.clip(density2, 0, 1)

    density3, origin3, step3 = pocket_density_from_mol_RAPID_Net_run3(mol)
    density3 = np.clip(density3, 0, 1)

    density4, origin4, step4 = pocket_density_from_mol_RAPID_Net_run4(mol)
    density4 = np.clip(density4, 0, 1)

    density5, origin5, step5 = pocket_density_from_mol_RAPID_Net_run5(mol)
    density5 = np.clip(density5, 0, 1)

    pockets = minimal_pockets_segmentation(density1, density2, density3, density4, density5, threshold=0.5, min_size=50, scale=0.5)

    i = 0
    for pocket_label in range(1, pockets.max() + 1):
        indices = np.argwhere(pockets == pocket_label).astype('float')
        indices *= step1
        indices += origin1
        mol = openbabel.OBMol()
        for idx in indices:
            a = mol.NewAtom()
            a.SetVector(float(idx[0]), float(idx[1]), float(idx[2]))
        p_mol = pybel.Molecule(mol)
        # Need to modify this line
        p_mol.write(format, path + '/pocket_thr05_Minimal' + str(i) + '.' + format)
        i += 1


def count_atoms_in_pdb(file_path):
    """Counts atoms in a PDB file."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return sum(1 for line in lines if line.startswith("ATOM") or line.startswith("HETATM"))


def calculate_distance(atom1, atom2):
    """Calculates the distance between two atoms."""
    coord1 = np.array(atom1.get_coord())
    coord2 = np.array(atom2.get_coord())
    return np.linalg.norm(coord1 - coord2)


def is_within_distance(protein_atoms, pocket_file, distance_threshold=6.0):
    """Checks if any atom in the pocket file is within a given distance of the protein atoms."""
    parser = PDBParser(QUIET=True)
    pocket_structure = parser.get_structure("Pocket", pocket_file)

    for pocket_model in pocket_structure:
        for pocket_chain in pocket_model:
            for pocket_residue in pocket_chain:
                for pocket_atom in pocket_residue:
                    for protein_atom in protein_atoms:
                        if calculate_distance(protein_atom, pocket_atom) <= distance_threshold:
                            return True
    return False


def remove_invalid_pockets(protein_file, folder_path, distance_threshold=6.0):
    """Removes empty pockets and those beyond the distance threshold from the protein."""
    parser = PDBParser(QUIET=True)
    protein_structure = parser.get_structure("Protein", protein_file)

    # Extract all protein atoms
    protein_atoms = [atom for model in protein_structure
                     for chain in model
                     for residue in chain
                     for atom in residue]

    remaining_pocket_files = []

    for file_name in os.listdir(folder_path):
        if file_name.startswith("pocket") and file_name.endswith(".pdb"):
            pocket_file_path = os.path.join(folder_path, file_name)

            # Check if the file is empty (zero atoms)
            if count_atoms_in_pdb(pocket_file_path) == 0:
                os.remove(pocket_file_path)
                print(f"Deleted empty pocket file: {file_name}")
                continue

            # Check if the file is within the required distance
            if not is_within_distance(protein_atoms, pocket_file_path, distance_threshold):
                os.remove(pocket_file_path)
                print(f"Removed distant pocket file: {file_name}")
            else:
                remaining_pocket_files.append(pocket_file_path)
                print(f"Retained pocket file: {file_name}")

    print("\nFinal list of retained pocket files:")
    for file in remaining_pocket_files:
        print(file)



