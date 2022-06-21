import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler
from models.gat import GATNet
from models.gat_gcn import GAT_GCN
from utils import *
import matplotlib.pyplot as plt
#from sampler import ImbalancedDatasetSampler


# training function at each epoch
def train(model, device, train_loader, optimizer, epoch):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    total_train = torch.Tensor()
    total_label = torch.Tensor()
    train_losses = []
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output,w = model(data)
        loss = loss_fn(output, data.y.view(-1,1).float()).to(device)
        loss = torch.mean(loss).float()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data.x),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))                                                                                                                                 	
    total_train = torch.cat((total_train, output.cpu()), 0)
    total_label = torch.cat((total_label, data.y.view(-1, 1).cpu()), 0)
    G_train = total_label.detach().numpy().flatten()
    P_train = total_train.detach().numpy().flatten()
    ret = [auc(G_train,P_train),pre(G_train,P_train),recall(G_train,P_train),f1(G_train,P_train),acc(G_train,P_train),mcc(G_train,P_train),spe(G_train,P_train)]
    print('train_auc',ret[0])
    print('train_pre',ret[1])
    print('train_recall',ret[2])
    print('train_f1',ret[3])
    print('train_acc',ret[4])
    print('train_mcc',ret[5])
    print('train_spe',ret[6])
    print('train_loss',np.average(train_losses))
    return G_train, P_train, np.average(train_losses)

def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    losses = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output,w = model(data)
            loss = loss_fn(output, data.y.view(-1,1).float())
            loss = torch.mean(loss).float().to(device)
            losses.append(loss.item())
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(),total_preds.numpy().flatten(),np.average(losses),w

modeling = [GATNet, GAT_GCN][int(sys.argv[1])]
model_st = modeling.__name__

cuda_name = "cuda:0"
print('cuda_name:', cuda_name)
    
TRAIN_BATCH_SIZE = 512
TEST_BATCH_SIZE = 512
LR = 0.0005
LOG_INTERVAL = 20
NUM_EPOCHS = 1000

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)

# Main program: iterate over different datasets
print('\nrunning on ', model_st + '_cyp')
processed_data_file_train = 'data/processed/cyp_train.pt'
processed_data_file_test_1a2 = 'data/processed/cyp_test_1a2.pt'
processed_data_file_valid = 'data/processed/cyp_valid.pt'
processed_data_file_test_2c9 = 'data/processed/cyp_test_2c9.pt'
processed_data_file_test_2c19 = 'data/processed/cyp_test_2c19.pt'
processed_data_file_test_2d6 = 'data/processed/cyp_test_2d6.pt'
processed_data_file_test_3a4 = 'data/processed/cyp_test_3a4.pt'
if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_valid))):
    print('please run create_data.py to prepare data in pytorch format!')
else:
    train_data = TestbedDataset(root='data', dataset='cyp_train')
    test_1a2_data = TestbedDataset(root='data', dataset='cyp_test_1a2')
    valid_data = TestbedDataset(root='data', dataset='cyp_valid')
    test_2c9_data = TestbedDataset(root='data', dataset='cyp_test_2c9')
    test_2c19_data = TestbedDataset(root='data', dataset='cyp_test_2c19')
    test_2d6_data = TestbedDataset(root='data', dataset='cyp_test_2d6')
    test_3a4_data = TestbedDataset(root='data', dataset='cyp_test_3a4')
    train_set = pd.read_csv('cyp_data/cyp_train.csv')
    lables_unique, counts = np.unique(train_set['score'],return_counts = True)
    class_weights = [sum(counts)/ c for c in counts]
    example_weights = [class_weights[e] for e in train_set['score']]
    sampler = WeightedRandomSampler(example_weights, len(train_set['score']))
    # make data PyTorch mini-batch processing ready
    #train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, sampler=sampler)
    train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=TEST_BATCH_SIZE, shuffle=False)
    test_1a2_loader = DataLoader(test_1a2_data, batch_size=TEST_BATCH_SIZE, shuffle=False)
    test_2c9_loader = DataLoader(test_2c9_data, batch_size=TEST_BATCH_SIZE, shuffle=False)
    test_2c19_loader = DataLoader(test_2c19_data, batch_size=TEST_BATCH_SIZE, shuffle=False)
    test_3a4_loader = DataLoader(test_3a4_data, batch_size=TEST_BATCH_SIZE,shuffle=False)
    test_2d6_loader = DataLoader(test_2d6_data, batch_size=TEST_BATCH_SIZE,shuffle=False)
    # training the model
    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    model = modeling().to(device)
    print(model)
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best_loss = 100
    best_test_auc = 1000
    best_test_ci = 0
    best_epoch = -1
    patience = 30
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    model_file_name = 'model_' + model_st + '_' + 'cyp.model'
    result_file_name = 'result_' + model_st + '_' + 'cyp.csv'
    train_losses=[]
    train_accs=[]
    valid_losses=[]
    valid_accs=[]
    for epoch in range(NUM_EPOCHS):
        G_T,P_T,train_loss = train(model, device, train_loader, optimizer, epoch+1)
        print('predicting for valid data')
        G,P,loss_valid,w= predicting(model, device, valid_loader)
        loss_valid_value = loss_valid
        print('valid_loss',loss_valid)
        print('valid_auc',auc(G,P))
        print('valid_pre',pre(G,P))
        print('valid_recall',recall(G,P))
        print('valid_f1',f1(G,P))
        print('valid_acc',acc(G,P))
        print('valid_mcc',mcc(G,P))
        print('valid_spe',spe(G,P))
        train_losses.append(np.array(train_loss))
        valid_losses.append(loss_valid)
        train_accs.append(acc(G_T,P_T))
        valid_accs.append(acc(G,P))
        b = pd.DataFrame({'value':G,'prediction':P})
        names = 'model_'+'value_validation'+'.csv'
        b.to_csv(names,sep=',') 
        early_stopping(loss_valid, model)
        if early_stopping.early_stop:
            print("Early stopping")
            torch.save(model.state_dict(), model_file_name)
            print('predicting for test data')
            G,P,loss_test_1a2,w_1a2 = predicting(model, device, test_1a2_loader)
            ret_1a2 = [auc(G,P),pre(G,P),recall(G,P),f1(G,P),acc(G,P),mcc(G,P),spe(G,P)]
            print('cyp_1a2 ',best_epoch,'auc',ret_1a2[0],'pre',ret_1a2[1],'recall',ret_1a2[2],'f1',ret_1a2[3],'acc',ret_1a2[4],'mcc',ret_1a2[5],'spe',ret_1a2[6])
            G,P,loss_test_2c9,w_2c9 = predicting(model, device, test_2c9_loader)
            ret_2c9 = [auc(G,P),pre(G,P),recall(G,P),f1(G,P),acc(G,P),mcc(G,P),spe(G,P)]
            print('cyp_2c9 ',best_epoch,'auc',ret_2c9[0],'pre',ret_2c9[1],'recall',ret_2c9[2],'f1',ret_2c9[3],'acc',ret_2c9[4],'mcc',ret_2c9[5],'spe',ret_2c9[6])
            G,P,loss_test_2c19,w_2c19 = predicting(model, device, test_2c19_loader)
            ret_2c19 = [auc(G,P),pre(G,P),recall(G,P),f1(G,P),acc(G,P),mcc(G,P),spe(G,P)]
            print('cyp_2c19 ',best_epoch,'auc',ret_2c19[0],'pre',ret_2c19[1],'recall',ret_2c19[2],'f1',ret_2c19[3],'acc',ret_2c19[4],'mcc',ret_2c19[5],'spe',ret_2c19[6])
            G,P,loss_test_2d6,w_2d6 = predicting(model, device, test_2d6_loader)
            ret_2d6 = [auc(G,P),pre(G,P),recall(G,P),f1(G,P),acc(G,P),mcc(G,P),spe(G,P)]
            print('cyp__2d6 ',best_epoch,'auc',ret_2d6[0],'pre',ret_2d6[1],'recall',ret_2d6[2],'f1',ret_2d6[3],'acc',ret_2d6[4],'mcc',ret_2d6[5],'spe',ret_2d6[6])
            G,P,loss_test_3a4,w_3a4 = predicting(model, device, test_3a4_loader)
            ret_3a4 = [auc(G,P),pre(G,P),recall(G,P),f1(G,P),acc(G,P),mcc(G,P),spe(G,P)]
            print('cyp_3a4 ',best_epoch,'auc',ret_3a4[0],'pre',ret_3a4[1],'recall',ret_3a4[2],'f1',ret_3a4[3],'acc',ret_3a4[4],'mcc',ret_3a4[5],'spe',ret_3a4[6])
            a = pd.DataFrame({'value':G,'prediction':P})
            name = 'model_'+'value_test'+'.csv'
            a.to_csv(name,sep=',')
            break
        else:
            print('no early stopping')
    df = pd.DataFrame({'train_loss':train_losses,'valid_loss':valid_losses,'train_accs':train_accs,'valid_accs':valid_accs})
    names = 'model_'+'loss_acc'+'.csv'
    df.to_csv(names,sep=',')