# GCNN
This is a simple codes to train a cyp-inhibitor prediction model with GCN and CNN.

Fiirstly, 'python create_data.py' was used to transform SMILES to molecular graph on moleculars and encode AA sequences.

Then, 'python training_validation.py 1' was used to train model with GAT_GCN. Here, we also provide GAT model by command 'python training_validation.py 0'.
