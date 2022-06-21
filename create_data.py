import pandas as pd
import numpy as np
import os
import json,pickle
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx
from utils import *

def atom_features(atom, explicit_H = False, use_chirality=False):
    results = one_of_k_encoding_unk(
      atom.GetSymbol(), #11
      [
        'C',
        'N',
        'O',
        'F',
        'P',
        'S',
        'Si'
        'Cl',
        'Br',
        'I',
        'other'
      ]) + one_of_k_encoding(atom.GetDegree(),
                             [0,1, 2, 3, 4 ,5]) + \
              one_of_k_encoding_unk(atom.GetFormalCharge(),[-1,0,1])+ one_of_k_encoding(atom.GetExplicitValence() ,[0,1,2,3,4,5,6]) + \
              one_of_k_encoding_unk(atom.GetHybridization(), [
                Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3]) + [atom.GetIsAromatic()]
    if not explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                  [0, 1, 2, 3, 4 ])
    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
                atom.GetProp('_CIPCode'),
                ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False
                                 ] + [atom.HasProp('_ChiralityPossible')]

    return np.array(results)
    

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    
    c_size = mol.GetNumAtoms()
    
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append( feature / sum(feature) )

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
        
    return c_size,features,edge_index


def seq_cat(prot):
    x = np.zeros(max_seq_len) 
    for i, ch in enumerate(prot[:max_seq_len]): 
        x[i] = seq_dict[ch]
    return x  


seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
seq_dict = {v:(i+1) for i,v in enumerate(seq_voc)}
seq_dict_len = len(seq_dict)
max_seq_len = 42


compound_iso_smiles = []
opts = ['train','test_1a2','valid','test_2c9','test_2c19','test_2d6','test_3a4']
for opt in opts:
        df = pd.read_csv('cyp_data/cyp'+ '_' + opt + '.csv')
        compound_iso_smiles += list( df['compound'] )
compound_iso_smiles = set(compound_iso_smiles)
smile_graph = {}
for smile in compound_iso_smiles:
    g = smile_to_graph(smile) 
    smile_graph[smile] = g

# convert to PyTorch data format
processed_data_file_train = 'data/processed/cyp_train.pt'
processed_data_file_test_1a2 = 'data/processed/cyp_test_1a2.pt'
processed_data_file_valid = 'data/processed/cyp_valid.pt'
processed_data_file_test_2c9 = 'data/processed/cyp_test_2c9.pt'
processed_data_file_test_2c19 = 'data/processed/cyp_test_2c19.pt'
processed_data_file_test_2d6 = 'data/processed/cyp_test_2d6.pt'
processed_data_file_test_3a4 = 'data/processed/cyp_test_3a4.pt'
if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_valid))):
    df = pd.read_csv('cyp_data/cyp_train.csv')
    train_drugs, train_prots,  train_Y = list(df['compound']),list(df['target_sequence']),list(df['score'])
    XT = [seq_cat(t) for t in train_prots]
    train_drugs, train_prots,  train_Y = np.asarray(train_drugs), np.asarray(XT), np.asarray(train_Y)
    df = pd.read_csv('cyp_data/cyp_test_1a2.csv')
    test_1a2_drugs, test_1a2_prots,  test_1a2_Y = list(df['compound']),list(df['target_sequence']),list(df['score'])
    XT = [seq_cat(t) for t in test_1a2_prots]
    test_1a2_drugs, test_1a2_prots,  test_1a2_Y = np.asarray(test_1a2_drugs), np.asarray(XT), np.asarray(test_1a2_Y)
    df = pd.read_csv('cyp_data/cyp_valid.csv')
    valid_drugs, valid_prots,  valid_Y = list(df['compound']),list(df['target_sequence']),list(df['score'])
    XT = [seq_cat(t) for t in valid_prots]
    valid_drugs, valid_prots,  valid_Y = np.asarray(valid_drugs), np.asarray(XT), np.asarray(valid_Y)
    df = pd.read_csv('cyp_data/cyp_test_2c9.csv')
    test_2c9_drugs, test_2c9_prots,  test_2c9_Y = list(df['compound']),list(df['target_sequence']),list(df['score'])
    XT = [seq_cat(t) for t in test_2c9_prots]
    test_2c9_drugs, test_2c9_prots,  test_2c9_Y = np.asarray(test_2c9_drugs), np.asarray(XT), np.asarray(test_2c9_Y)
    df = pd.read_csv('cyp_data/cyp_test_2c19.csv')
    test_2c19_drugs, test_2c19_prots,  test_2c19_Y = list(df['compound']),list(df['target_sequence']),list(df['score'])
    XT = [seq_cat(t) for t in test_2c19_prots]
    test_2c19_drugs, test_2c19_prots,  test_2c19_Y = np.asarray(test_2c19_drugs), np.asarray(XT), np.asarray(test_2c19_Y)
    df = pd.read_csv('cyp_data/cyp_test_2d6.csv')
    test_2d6_drugs, test_2d6_prots,  test_2d6_Y = list(df['compound']),list(df['target_sequence']),list(df['score'])
    XT = [seq_cat(t) for t in test_2d6_prots]
    test_2d6_drugs, test_2d6_prots,  test_2d6_Y = np.asarray(test_2d6_drugs), np.asarray(XT), np.asarray(test_2d6_Y)
    df = pd.read_csv('cyp_data/cyp_test_3a4.csv')
    test_3a4_drugs, test_3a4_prots,  test_3a4_Y = list(df['compound']),list(df['target_sequence']),list(df['score'])
    XT = [seq_cat(t) for t in test_3a4_prots]
    test_3a4_drugs, test_3a4_prots,  test_3a4_Y = np.asarray(test_3a4_drugs), np.asarray(XT), np.asarray(test_3a4_Y)
    # make data PyTorch Geometric ready
    print('preparing ', 'cyp_train.pt in pytorch format!')
    train_data = TestbedDataset(root='data', dataset='cyp_train', xd=train_drugs, xt=train_prots, y=train_Y,smile_graph=smile_graph)
    print('preparing ', 'cyp_valid.pt in pytorch format!')
    valid_data = TestbedDataset(root='data', dataset='cyp_valid', xd=valid_drugs, xt=valid_prots, y=valid_Y,smile_graph=smile_graph)
    print('preparing ', 'cyp_test_1a2.pt in pytorch format!')
    test_1a2_data = TestbedDataset(root='data', dataset='cyp_test_1a2', xd=test_1a2_drugs, xt=test_1a2_prots, y=test_1a2_Y,smile_graph=smile_graph)
    print('preparing ', 'cyp_test_2c9.pt in pytorch format!')
    test_2c9_data = TestbedDataset(root='data', dataset='cyp_test_2c9', xd=test_2c9_drugs, xt=test_2c9_prots, y=test_2c9_Y,smile_graph=smile_graph)
    print('preparing ', 'cyp_test_2c19.pt in pytorch format!')
    test_2c19_data = TestbedDataset(root='data', dataset='cyp_test_2c19', xd=test_2c19_drugs, xt=test_2c19_prots, y=test_2c19_Y,smile_graph=smile_graph)
    print('preparing ', 'cyp_test_2d6.pt in pytorch format!')
    test_2d6_data = TestbedDataset(root='data', dataset='cyp_test_2d6', xd=test_2d6_drugs, xt=test_2d6_prots, y=test_2d6_Y,smile_graph=smile_graph)
    print('preparing ', 'cyp_test_3a4.pt in pytorch format!')
    test_3a4_data = TestbedDataset(root='data', dataset='cyp_test_3a4', xd=test_3a4_drugs, xt=test_3a4_prots, y=test_3a4_Y,smile_graph=smile_graph)
    print(processed_data_file_train, ' and ', processed_data_file_test_1a2,'and', processed_data_file_valid,'and so on', 'have been created')
else:
    print(processed_data_file_train,' and ',processed_data_file_valid, ' are already created')        
