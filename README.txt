Roi Papo
Chen Pery

first you need to clone:
https://github.com/LiheYoung/Depth-Anything
and install requirements

than paste the python files in our code folder into the cloned folder

to reproduce the results of our research work you will need:
the animals dataset:
https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals

to reproduce the paper results you will need:
the nyu2 dataset:
https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html

also you will need to get the checkpoint (weights) from the git above


Code Structure:
classification_utils.py- metrics for evaluating our classification models

data_parser.py- the code that is responsible for parsing the animals dataset, splitting and creating train val sets

resnet_extractor.py- preprocesses images and uses a pre-trained ResNet-50 model to extract feature vectors. It loads a given dataset, and replaces the model's final layer with an identity layer. The script then processes the images through the model to obtain feature vectors, which are shuffled and saved as .npy files.

classifier.py script loads pre-saved feature vectors and labels from .npy files, preprocesses the data, and creates DataLoader objects. It defines a multi-layer perceptron (MLP) model for classification and sets up the training process (for our 3 experiments) using PyTorch. The script trains the MLP model on the training data and evaluates it on the validation data, logging the results. Finally, it uses wandb for experiment tracking and hyperparameter optimization.

run.py - we changed the original run.py file of the authors to make it custom for depth inference animal dataset while keeping train val structure in order produce the experiment datasets  