import os
import shutil
from sklearn.model_selection import train_test_split
import random

def create_dataset_structure(base_dir, new_base_dir):
    # Create new directories
    train_dir = os.path.join(new_base_dir, 'train')
    val_dir = os.path.join(new_base_dir, 'validation')
    
    if not os.path.exists(new_base_dir):
        os.makedirs(new_base_dir)
    
    for animal_class in os.listdir(base_dir):
        class_path = os.path.join(base_dir, animal_class)
        if os.path.isdir(class_path):
            os.makedirs(os.path.join(train_dir, animal_class), exist_ok=True)
            os.makedirs(os.path.join(val_dir, animal_class), exist_ok=True)

def split_and_copy_images(base_dir, new_base_dir, test_size=0.2, random_state=42):
    train_dir = os.path.join(new_base_dir, 'train')
    val_dir = os.path.join(new_base_dir, 'validation')
    
    for animal_class in os.listdir(base_dir):
        class_path = os.path.join(base_dir, animal_class)
        if os.path.isdir(class_path):
            images = os.listdir(class_path)
            train_images, val_images = train_test_split(images, test_size=test_size, random_state=random_state)
            
            for image in train_images:
                src = os.path.join(class_path, image)
                dst = os.path.join(train_dir, animal_class, image)
                shutil.copy(src, dst)
            
            for image in val_images:
                src = os.path.join(class_path, image)
                dst = os.path.join(val_dir, animal_class, image)
                shutil.copy(src, dst)

def main():
    # Set the random seed for reproducibility
    random.seed(42)
    
    # Define the base directory for the original dataset
    base_dir = '/data/home/roipapo/Depth-Anything/depth_vis'
    
    # Define the base directory for the new dataset
    new_base_dir = 'depth_animals'
    
    # Create the dataset structure
    create_dataset_structure(base_dir, new_base_dir)
    
    # Split the images and copy them to the new dataset structure
    split_and_copy_images(base_dir, new_base_dir)

if __name__ == "__main__":
    main()
