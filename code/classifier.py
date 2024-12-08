import os
from collections import Counter
import random
import copy
import torch
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import random
import time
from tqdm import tqdm
import wandb
from classification_utils import calculate_pred
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def get_optimizer(model, code: str):
    if code == "SGD":
        return optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-3)
    if code == "adam":
        return optim.Adam(model.parameters(), lr=0.001)
    if code == "adamW":
        return optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-3)


def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=100,experiment_str=None):
    """Responsible for running the training and validation phases for the requested model."""
    since = time.time()
    
   
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    loss_dict = {'train': [], 'val': []}
    acc_dict = {'train': [], 'val': []}

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train() 
            else:
                model.eval()   

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.squeeze(1).long().to(device)

                optimizer.zero_grad()

                # track history only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            acc_dict[phase].append(epoch_acc.item())
            loss_dict[phase].append(epoch_loss)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            if phase == 'val':
                wandb.log({'val loss': epoch_loss, 'val accuracy': epoch_acc})
            if phase == 'train':
                wandb.log({'train loss': epoch_loss, 'train accuracy': epoch_acc})

            # If the current epoch provides the best validation accuracy so far, save the model's weights.
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, loss_dict, acc_dict


def tree_fitter_fv(x_train, y_train, x_val, y_val):
    params = {
        'min_child_weight': [0.5, 1, 5, 10],
        'gamma': [0.1, 0.5, 1, 1.5, 2, 5],
        'subsample': [0.4, 0.6, 0.8, 1.0],
        'reg_alpha': [0, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4,
                      12.8, 25.6, 51.2, 102.4, 200],
        'reg_lambda': [0, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 25.6, 51.2, 102.4, 200],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
    }
    xgb = XGBClassifier(learning_rate=0.001, n_estimators=400, objective='binary:logistic', eval_metric='error',
                        nthread=4)
    folds = 3
    param_comb = 5

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1001)

    random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb, scoring='roc_auc', n_jobs=4,
                                       cv=skf.split(x_train, y_train), random_state=1001)

    # Here we go
    random_search.fit(x_train, y_train)
    results = pd.DataFrame(random_search.cv_results_)
    results.to_csv('xgb-random-grid-search-results-01.csv', index=False)
    predicted_y = random_search.best_estimator_.predict(x_val)
    proba = random_search.best_estimator_.predict_proba(x_val)
    print("Tree:")
    calculate_pred(pred=predicted_y, y_val=y_val)
    print("...................")
    return proba


import numpy as np

# Load training feature vectors and labels
depth_train_fvecs = np.load('Resnet_on_Normal_depth_animals_train_fvecs_2024-06-05.npy')
depth_train_labels = np.load('Resnet_on_Normal_depth_animals_train_lables_2024-06-05.npy')

splitted_train_fvecs = np.load('Resnet_on_Normal_splitted_animals_train_fvecs_2024-06-05.npy')
splitted_train_labels = np.load('Resnet_on_Normal_splitted_animals_train_lables_2024-06-05.npy')

# Load validation feature vectors and labels
depth_val_fvecs = np.load('Resnet_on_Normal_depth_animals_val_fvecs_2024-06-05.npy')
depth_val_labels = np.load('Resnet_on_Normal_depth_animals_val_lables_2024-06-05.npy')

splitted_val_fvecs = np.load('Resnet_on_Normal_splitted_animals_val_fvecs_2024-06-05.npy')
splitted_val_labels = np.load('Resnet_on_Normal_splitted_animals_val_lables_2024-06-05.npy')

# Concatenate training feature vectors

class MLPClassifier(nn.Module):
    def __init__(self, input_size=1000,drop_oup=0.5, hidden_size=512, num_classes=90):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(p=drop_oup)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        return out



def MLP_fitter(x_train, y_train, x_val, y_val,sweep_config):
    # task = Task.init(project_name='Ensemble', task_name=name_of_run)
    # logger = task.get_logger()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Normalize input data
    # np.where(np.isnan(x_train[:, -1]))
    for set in [x_train, x_val]:
        column_means = np.nanmean(set, axis=0)
        for i in range(set.shape[1]):
            column = set[:, i]
            column[np.isnan(column)] = column_means[i]

    # np.where(np.isnan(x_train[:,-1]))
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)

    x_train = (x_train - mean) / (std + 1e-8)  # Adding a small constant to avoid division by zero
    x_val = (x_val - mean) / (std + 1e-8)

    # Convert numpy arrays to PyTorch tensors
    x_train = torch.FloatTensor(x_train).to(device)
    y_train = torch.FloatTensor(y_train).unsqueeze(1).to(device)  # Add a dimension for compatibility
    x_val = torch.FloatTensor(x_val).to(device)
    y_val = torch.FloatTensor(y_val).unsqueeze(1).to(device)

    # Create DataLoader for training and validation data
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=sweep_config.batch_size, shuffle=False)

    val_dataset = TensorDataset(x_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=sweep_config.batch_size, shuffle=False)
    dataloaders = {'train': train_loader, 'val': val_loader}
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

    # Initialize the model, loss function, and optimizer
    input_size = 2000 if sweep_config.mode == "all" else 1000
    model = MLPClassifier(input_size=input_size,drop_oup=sweep_config.drop_out).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model,sweep_config.optimizer)
    # optimizer = optim.Adam(model.parameters(), lr=0.001)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)

    train_model(model=model,criterion=criterion,optimizer=optimizer,scheduler=scheduler,dataloaders=dataloaders,dataset_sizes=dataset_sizes,num_epochs=30,experiment_str=sweep_config.mode)

# model = models.resnet50(pretrained=False)
# head=model.fc


sweep_configuration = {
    'method': 'grid',
    'metric':
        {
            'goal': 'maximize',
            'name': 'accuracy'
        },
    'parameters':
        {
    'drop_out': {'values': [0.6]},
    'batch_size':{'values': [250]},
    'optimizer': {'values': ['adamW']},
    'mode':  {'values': ["depth","splitted","all"]},
        }
}
 
sweep_id = wandb.sweep(
    sweep=sweep_configuration,
    project='Fvecs_classification'
)
# run = wandb.init(project=f"classifing_Fvecs", name=f"run_{mode}", reinit=True)

# modes= ["depth","splitted","all"]
def main(config=None):
    with wandb.init(config=config):
        config = wandb.config
        print(config)
        mode = config.mode
        print(f"######## MODE is {mode} ##########")
        # print(mode)
        if mode == "all":
            x_train = np.hstack((depth_train_fvecs, splitted_train_fvecs))
            # Concatenate validation feature vectors
            x_val = np.hstack((depth_val_fvecs, splitted_val_fvecs))
        if mode == "depth":
            x_train = depth_train_fvecs
            # Concatenate validation feature vectors
            x_val = depth_val_fvecs
        if mode == "splitted":
            x_train = splitted_train_fvecs
            # Concatenate validation feature vectors
            x_val =  splitted_val_fvecs
            # Ensure that the labels are the same for depth and splitted
        assert np.array_equal(depth_train_labels, splitted_train_labels), "Training labels do not match"
        assert np.array_equal(depth_val_labels, splitted_val_labels), "Validation labels do not match"

        # Use the labels from either depth or splitted (they are the same)
        y_train = depth_train_labels
        y_val = depth_val_labels

        MLP_fitter(x_train, y_train, x_val, y_val, sweep_config=config)


        # Call the function
        # tree_fitter_fv(x_train, y_train, x_val, y_val)

wandb.agent(sweep_id, function=main)
