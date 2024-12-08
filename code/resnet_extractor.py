import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory
# import wandb
from datetime import date



def today():
    from datetime import date

    today = date.today()
    date_string = today.strftime("%Y-%m-%d")

    return date_string


data_transforms = {
'train': transforms.Compose([
    transforms.Resize([300,300]),
    transforms.RandomRotation((-20, 20)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]),
'val': transforms.Compose([
    transforms.Resize([300,300]),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]),
}


image_sets=["depth","splitted"]
shuffled_indices={"train": None, "val":None}

for im_set in image_sets:
    data_dir = os.path.join(f"{im_set}_animals")
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                    for x in ['train', 'val']}


    experiment_name= f"Resnet_on_Normal_{data_dir}"

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=20,
                                                shuffle=False, num_workers=4)
                for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torchvision.models.resnet50(pretrained=True)



    datasets_types = ["train","val"]
    for ds_type in datasets_types:
        
        images_order = dataloaders[ds_type].dataset.samples
        img_ids_order = [os.path.basename(img_id[0])[1:-4].split('_')[0] for img_id in images_order]
        labels = [img_id[1] for img_id in images_order]
        model.head = nn.Identity()
        infer_df = []
        feature_vectors = []

        model.eval()
        model.to(device)
        

        image_names = []
        dataset = dataloaders[ds_type].dataset
        for inputs, labels2 in dataloaders[ds_type]:
            inputs = inputs.to(device)
            outputs = model(inputs)
            # Get the list of image paths from the dataset samples
            image_paths = [sample[0] for sample in dataset.samples]

            # Append the image paths to the list
            image_names.extend(image_paths)
            feature_vectors.append(outputs.cpu().detach().numpy())
        with torch.no_grad():
            feature_vectors_ndarray = np.concatenate(feature_vectors, axis=0)
            if shuffled_indices[ds_type] is None:
                shuffled_indices[ds_type] = np.random.permutation(len(feature_vectors_ndarray))
            shuffled_features = feature_vectors_ndarray[shuffled_indices[ds_type]]
            shuffled_lables = np.array(labels)[shuffled_indices[ds_type]]
            np.save(f'{experiment_name}_{ds_type}_lables_{today()}.xnpy', shuffled_lables)
            np.save(f'{experiment_name}_{ds_type}_fvecs_{today()}.xnpy', shuffled_features)
            print(f"features of {ds_type} saved")
