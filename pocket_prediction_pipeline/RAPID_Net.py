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
    """Count atoms in PDB file to erase empty pockets"""
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


# Featurizer from tfbio package
# https://gitlab.com/cheminfIBB/tfbio
import os
import numpy as np
import py3Dmol
import scipy.stats as stats
import scipy.cluster.hierarchy
from scipy.spatial.distance import squareform
from scipy.interpolate import griddata

import matplotlib.pyplot as plt
import seaborn as sb
from matplotlib import colors

import pandas as pd
from biopandas.pdb import PandasPdb

from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdMolTransforms, rdDepictor, rdForceFieldHelpers
from rdkit.Chem.Draw import rdMolDraw2D, IPythonConsole
from IPython.display import SVG, Image
import ipywidgets as widgets

from openbabel import pybel

import pickle
from math import ceil, sin, cos, sqrt, pi
from itertools import combinations

from statistics import mean, stdev



class Featurizer():
    """Calcaulates atomic features for molecules. Features can encode atom type,
    native pybel properties or any property defined with SMARTS patterns

    Attributes
    ----------
    FEATURE_NAMES: list of strings
        Labels for features (in the same order as features)
    NUM_ATOM_CLASSES: int
        Number of atom codes
    ATOM_CODES: dict
        Dictionary mapping atomic numbers to codes
    NAMED_PROPS: list of string
        Names of atomic properties to retrieve from pybel.Atom object
    CALLABLES: list of callables
        Callables used to calculcate custom atomic properties
    SMARTS: list of SMARTS strings
        SMARTS patterns defining additional atomic properties
    """

    def __init__(self, atom_codes=None, atom_labels=None,
                 named_properties=None, save_molecule_codes=True,
                 custom_properties=None, smarts_properties=None,
                 smarts_labels=None):

        """Creates Featurizer with specified types of features. Elements of a
        feature vector will be in a following order: atom type encoding
        (defined by atom_codes), Pybel atomic properties (defined by
        named_properties), molecule code (if present), custom atomic properties
        (defined `custom_properties`), and additional properties defined with
        SMARTS (defined with `smarts_properties`).

        Parameters
        ----------
        atom_codes: dict, optional
            Dictionary mapping atomic numbers to codes. It will be used for
            one-hot encoging therefore if n different types are used, codes
            shpuld be from 0 to n-1. Multiple atoms can have the same code,
            e.g. you can use {6: 0, 7: 1, 8: 1} to encode carbons with [1, 0]
            and nitrogens and oxygens with [0, 1] vectors. If not provided,
            default encoding is used.
        atom_labels: list of strings, optional
            Labels for atoms codes. It should have the same length as the
            number of used codes, e.g. for `atom_codes={6: 0, 7: 1, 8: 1}` you
            should provide something like ['C', 'O or N']. If not specified
            labels 'atom0', 'atom1' etc are used. If `atom_codes` is not
            specified this argument is ignored.
        named_properties: list of strings, optional
            Names of atomic properties to retrieve from pybel.Atom object. If
            not specified ['hyb', 'heavyvalence', 'heterovalence',
            'partialcharge'] is used.
        save_molecule_codes: bool, optional (default True)
            If set to True, there will be an additional feature to save
            molecule code. It is usefeul when saving molecular complex in a
            single array.
        custom_properties: list of callables, optional
            Custom functions to calculate atomic properties. Each element of
            this list should be a callable that takes pybel.Atom object and
            returns a float. If callable has `__name__` property it is used as
            feature label. Otherwise labels 'func<i>' etc are used, where i is
            the index in `custom_properties` list.
        smarts_properties: list of strings, optional
            Additional atomic properties defined with SMARTS patterns. These
            patterns should match a single atom. If not specified, deafult
            patterns are used.
        smarts_labels: list of strings, optional
            Labels for properties defined with SMARTS. Should have the same
            length as `smarts_properties`. If not specified labels 'smarts0',
            'smarts1' etc are used. If `smarts_properties` is not specified
            this argument is ignored.
        """

        # Remember namse of all features in the correct order
        self.FEATURE_NAMES = []

        if atom_codes is not None:
            if not isinstance(atom_codes, dict):
                raise TypeError('Atom codes should be dict, got %s instead'
                                % type(atom_codes))
            codes = set(atom_codes.values())
            for i in range(len(codes)):
                if i not in codes:
                    raise ValueError('Incorrect atom code %s' % i)

            self.NUM_ATOM_CLASSES = len(codes)
            self.ATOM_CODES = atom_codes
            if atom_labels is not None:
                if len(atom_labels) != self.NUM_ATOM_CLASSES:
                    raise ValueError('Incorrect number of atom labels: '
                                     '%s instead of %s'
                                     % (len(atom_labels), self.NUM_ATOM_CLASSES))
            else:
                atom_labels = ['atom%s' % i for i in range(self.NUM_ATOM_CLASSES)]
            self.FEATURE_NAMES += atom_labels
        else:
            self.ATOM_CODES = {}

            metals = ([3, 4, 11, 12, 13] + list(range(19, 32))
                      + list(range(37, 51)) + list(range(55, 84))
                      + list(range(87, 104)))

            # List of tuples (atomic_num, class_name) with atom types to encode.
            atom_classes = [
                (5, 'B'),
                (6, 'C'),
                (7, 'N'),
                (8, 'O'),
                (15, 'P'),
                (16, 'S'),
                (34, 'Se'),
                ([9, 17, 35, 53], 'halogen'),
                (metals, 'metal')
            ]

            for code, (atom, name) in enumerate(atom_classes):
                if type(atom) is list:
                    for a in atom:
                        self.ATOM_CODES[a] = code
                else:
                    self.ATOM_CODES[atom] = code
                self.FEATURE_NAMES.append(name)

            self.NUM_ATOM_CLASSES = len(atom_classes)

        if named_properties is not None:
            if not isinstance(named_properties, (list, tuple, np.ndarray)):
                raise TypeError('named_properties must be a list')
            allowed_props = [prop for prop in dir(pybel.Atom)
                             if not prop.startswith('__')]
            for prop_id, prop in enumerate(named_properties):
                if prop not in allowed_props:
                    raise ValueError(
                        'named_properties must be in pybel.Atom attributes,'
                        ' %s was given at position %s' % (prop_id, prop)
                    )
            self.NAMED_PROPS = named_properties
        else:
            # pybel.Atom properties to save
            self.NAMED_PROPS = ['hyb', 'heavydegree', 'heterodegree',
                                'partialcharge']
        self.FEATURE_NAMES += self.NAMED_PROPS

        if not isinstance(save_molecule_codes, bool):
            raise TypeError('save_molecule_codes should be bool, got %s '
                            'instead' % type(save_molecule_codes))
        self.save_molecule_codes = save_molecule_codes
        if save_molecule_codes:
            # Remember if an atom belongs to the ligand or to the protein
            self.FEATURE_NAMES.append('molcode')

        self.CALLABLES = []
        if custom_properties is not None:
            for i, func in enumerate(custom_properties):
                if not callable(func):
                    raise TypeError('custom_properties should be list of'
                                    ' callables, got %s instead' % type(func))
                name = getattr(func, '__name__', '')
                if name == '':
                    name = 'func%s' % i
                self.CALLABLES.append(func)
                self.FEATURE_NAMES.append(name)

        if smarts_properties is None:
            # SMARTS definition for other properties
            self.SMARTS = [
                '[#6+0!$(*~[#7,#8,F]),SH0+0v2,s+0,S^3,Cl+0,Br+0,I+0]',
                '[a]',
                '[!$([#1,#6,F,Cl,Br,I,o,s,nX3,#7v5,#15v5,#16v4,#16v6,*+1,*+2,*+3])]',
                '[!$([#6,H0,-,-2,-3]),$([!H0;#7,#8,#9])]',
                '[r]'
            ]
            smarts_labels = ['hydrophobic', 'aromatic', 'acceptor', 'donor',
                             'ring']
        elif not isinstance(smarts_properties, (list, tuple, np.ndarray)):
            raise TypeError('smarts_properties must be a list')
        else:
            self.SMARTS = smarts_properties

        if smarts_labels is not None:
            if len(smarts_labels) != len(self.SMARTS):
                raise ValueError('Incorrect number of SMARTS labels: %s'
                                 ' instead of %s'
                                 % (len(smarts_labels), len(self.SMARTS)))
        else:
            smarts_labels = ['smarts%s' % i for i in range(len(self.SMARTS))]

        # Compile patterns
        self.compile_smarts()
        self.FEATURE_NAMES += smarts_labels

    def compile_smarts(self):
        self.__PATTERNS = []
        for smarts in self.SMARTS:
            self.__PATTERNS.append(pybel.Smarts(smarts))

    def encode_num(self, atomic_num):
        """Encode atom type with a binary vector. If atom type is not included in
        the `atom_classes`, its encoding is an all-zeros vector.

        Parameters
        ----------
        atomic_num: int
            Atomic number

        Returns
        -------
        encoding: np.ndarray
            Binary vector encoding atom type (one-hot or null).
        """

        if not isinstance(atomic_num, int):
            raise TypeError('Atomic number must be int, %s was given'
                            % type(atomic_num))

        encoding = np.zeros(self.NUM_ATOM_CLASSES)
        try:
            encoding[self.ATOM_CODES[atomic_num]] = 1.0
        except:
            pass
        return encoding

    def find_smarts(self, molecule):
        """Find atoms that match SMARTS patterns.

        Parameters
        ----------
        molecule: pybel.Molecule

        Returns
        -------
        features: np.ndarray
            NxM binary array, where N is the number of atoms in the `molecule`
            and M is the number of patterns. `features[i, j]` == 1.0 if i'th
            atom has j'th property
        """

        if not isinstance(molecule, pybel.Molecule):
            raise TypeError('molecule must be pybel.Molecule object, %s was given'
                            % type(molecule))

        features = np.zeros((len(molecule.atoms), len(self.__PATTERNS)))

        for (pattern_id, pattern) in enumerate(self.__PATTERNS):
            atoms_with_prop = np.array(list(*zip(*pattern.findall(molecule))),
                                       dtype=int) - 1
            features[atoms_with_prop, pattern_id] = 1.0
        return features

    def get_features(self, molecule, molcode=None):
        """Get coordinates and features for all heavy atoms in the molecule.

        Parameters
        ----------
        molecule: pybel.Molecule
        molcode: float, optional
            Molecule type. You can use it to encode whether an atom belongs to
            the ligand (1.0) or to the protein (-1.0) etc.

        Returns
        -------
        coords: np.ndarray, shape = (N, 3)
            Coordinates of all heavy atoms in the `molecule`.
        features: np.ndarray, shape = (N, F)
            Features of all heavy atoms in the `molecule`: atom type
            (one-hot encoding), pybel.Atom attributes, type of a molecule
            (e.g protein/ligand distinction), and other properties defined with
            SMARTS patterns
        """

        if not isinstance(molecule, pybel.Molecule):
            raise TypeError('molecule must be pybel.Molecule object,'
                            ' %s was given' % type(molecule))
        if molcode is None:
            if self.save_molecule_codes is True:
                raise ValueError('save_molecule_codes is set to True,'
                                 ' you must specify code for the molecule')
        elif not isinstance(molcode, (float, int)):
            raise TypeError('motlype must be float, %s was given'
                            % type(molcode))

        coords = []
        features = []
        heavy_atoms = []

        for i, atom in enumerate(molecule):
            # ignore hydrogens and dummy atoms (they have atomicnum set to 0)
            if atom.atomicnum > 1:
                heavy_atoms.append(i)
                coords.append(atom.coords)

                features.append(np.concatenate((
                    self.encode_num(atom.atomicnum),
                    [atom.__getattribute__(prop) for prop in self.NAMED_PROPS],
                    [func(atom) for func in self.CALLABLES],
                )))

        coords = np.array(coords, dtype=np.float32)
        features = np.array(features, dtype=np.float32)
        if self.save_molecule_codes:
            features = np.hstack((features,
                                  molcode * np.ones((len(features), 1))))
        features = np.hstack([features,
                              self.find_smarts(molecule)[heavy_atoms]])

        if np.isnan(features).any():
            raise RuntimeError('Got NaN when calculating features')

        return coords, features

    def to_pickle(self, fname='featurizer.pkl'):
        """Save featurizer in a given file. Featurizer can be restored with
        `from_pickle` method.

        Parameters
        ----------
        fname: str, optional
           Path to file in which featurizer will be saved
        """

        # patterns can't be pickled, we need to temporarily remove them
        patterns = self.__PATTERNS[:]
        del self.__PATTERNS
        try:
            with open(fname, 'wb') as f:
                pickle.dump(self, f)
        finally:
            self.__PATTERNS = patterns[:]

    @staticmethod
    def from_pickle(fname):
        """Load pickled featurizer from a given file

        Parameters
        ----------
        fname: str, optional
           Path to file with saved featurizer

        Returns
        -------
        featurizer: Featurizer object
           Loaded featurizer
        """
        with open(fname, 'rb') as f:
            featurizer = pickle.load(f)
        featurizer.compile_smarts()
        return featurizer


def rotation_matrix(axis, theta):
    """Counterclockwise rotation about a given axis by theta radians"""

    if not isinstance(axis, (np.ndarray, list, tuple)):
        raise TypeError('axis must be an array of floats of shape (3,)')
    try:
        axis = np.asarray(axis, dtype=np.float64)
    except ValueError:
        raise ValueError('axis must be an array of floats of shape (3,)')

    if axis.shape != (3,):
        raise ValueError('axis must be an array of floats of shape (3,)')

    if not isinstance(theta, (float, int)):
        raise TypeError('theta must be a float')

    axis = axis / sqrt(np.dot(axis, axis))
    a = cos(theta / 2.0)
    b, c, d = -axis * sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


# Create matrices for all possible 90* rotations of a box
ROTATIONS = [rotation_matrix([1, 1, 1], 0)]

# about X, Y and Z - 9 rotations
for a1 in range(3):
    for t in range(1, 4):
        axis = np.zeros(3)
        axis[a1] = 1
        theta = t * pi / 2.0
        ROTATIONS.append(rotation_matrix(axis, theta))

# about each face diagonal - 6 rotations
for (a1, a2) in combinations(range(3), 2):
    axis = np.zeros(3)
    axis[[a1, a2]] = 1.0
    theta = pi
    ROTATIONS.append(rotation_matrix(axis, theta))
    axis[a2] = -1.0
    ROTATIONS.append(rotation_matrix(axis, theta))

# about each space diagonal - 8 rotations
for t in [1, 2]:
    theta = t * 2 * pi / 3
    axis = np.ones(3)
    ROTATIONS.append(rotation_matrix(axis, theta))
    for a1 in range(3):
        axis = np.ones(3)
        axis[a1] = -1
        ROTATIONS.append(rotation_matrix(axis, theta))


def rotate(coords, rotation):
    """Rotate coordinates by a given rotation

    Parameters
    ----------
    coords: array-like, shape (N, 3)
        Arrays with coordinates and features for each atoms.
    rotation: int or array-like, shape (3, 3)
        Rotation to perform. You can either select predefined rotation by
        giving its index or specify rotation matrix.

    Returns
    -------
    coords: np.ndarray, shape = (N, 3)
        Rotated coordinates.
    """

    global ROTATIONS

    if not isinstance(coords, (np.ndarray, list, tuple)):
        raise TypeError('coords must be an array of floats of shape (N, 3)')
    try:
        coords = np.asarray(coords, dtype=np.float64)
    except ValueError:
        raise ValueError('coords must be an array of floats of shape (N, 3)')
    shape = coords.shape
    if len(shape) != 2 or shape[1] != 3:
        raise ValueError('coords must be an array of floats of shape (N, 3)')

    if isinstance(rotation, int):
        if rotation >= 0 and rotation < len(ROTATIONS):
            return np.dot(coords, ROTATIONS[rotation])
        else:
            raise ValueError('Invalid rotation number %s!' % rotation)
    elif isinstance(rotation, np.ndarray) and rotation.shape == (3, 3):
        return np.dot(coords, rotation)

    else:
        raise ValueError('Invalid rotation %s!' % rotation)


# TODO: add make_grid variant for GPU

def make_grid(coords, features, grid_resolution=1.0, max_dist=10.0):
    """Convert atom coordinates and features represented as 2D arrays into a
    fixed-sized 3D box.

    Parameters
    ----------
    coords, features: array-likes, shape (N, 3) and (N, F)
        Arrays with coordinates and features for each atoms.
    grid_resolution: float, optional
        Resolution of a grid (in Angstroms).
    max_dist: float, optional
        Maximum distance between atom and box center. Resulting box has size of
        2*`max_dist`+1 Angstroms and atoms that are too far away are not
        included.

    Returns
    -------
    coords: np.ndarray, shape = (M, M, M, F)
        4D array with atom properties distributed in 3D space. M is equal to
        2 * `max_dist` / `grid_resolution` + 1
    """

    try:
        coords = np.asarray(coords, dtype=np.float64)
    except ValueError:
        raise ValueError('coords must be an array of floats of shape (N, 3)')
    c_shape = coords.shape
    if len(c_shape) != 2 or c_shape[1] != 3:
        raise ValueError('coords must be an array of floats of shape (N, 3)')

    N = len(coords)
    try:
        features = np.asarray(features, dtype=np.float64)
    except ValueError:
        raise ValueError('features must be an array of floats of shape (N, F)')
    f_shape = features.shape
    if len(f_shape) != 2 or f_shape[0] != N:
        raise ValueError('features must be an array of floats of shape (N, F)')

    if not isinstance(grid_resolution, (float, int)):
        raise TypeError('grid_resolution must be float')
    if grid_resolution <= 0:
        raise ValueError('grid_resolution must be positive')

    if not isinstance(max_dist, (float, int)):
        raise TypeError('max_dist must be float')
    if max_dist <= 0:
        raise ValueError('max_dist must be positive')

    num_features = f_shape[1]
    max_dist = float(max_dist)
    grid_resolution = float(grid_resolution)

    box_size = ceil(2 * max_dist / grid_resolution + 1)

    # move all atoms to the neares grid point
    grid_coords = (coords + max_dist) / grid_resolution
    grid_coords = grid_coords.round().astype(int)

    # remove atoms outside the box
    in_box = ((grid_coords >= 0) & (grid_coords < box_size)).all(axis=1)
    grid = np.zeros((1, box_size, box_size, box_size, num_features),
                    dtype=np.float32)
    for (x, y, z), f in zip(grid_coords[in_box], features[in_box]):
        grid[0, x, y, z] += f

    return grid

featurizer = Featurizer(save_molecule_codes = False)

scale=0.5
max_dist=35
file_format = 'pdb'

grid_resolution=1.0
scale=0.5
grid_size=36

threshold = 0.5
resolution = 1. / scale
