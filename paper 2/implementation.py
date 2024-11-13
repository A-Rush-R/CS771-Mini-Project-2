import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision import models, transforms
import numpy as np
from RandMix import RandMix, AdaIN2d  # Import RandMix and AdaIN classes
from PCA import PCLoss, PCALoss
from LWPwithRaTP import LWPWithRaTP, ResNetFeatureExtractor

# Assuming that LWPWithRaTP and required dependencies are already defined in your code.
# Adjust arguments based on your requirements
class Args:
    # Set necessary arguments here, adjust them as needed
    lr = 0.01
    weight_decay = 1e-4
    num_classes = 10  # example
    proj_dim = {f'D{i}': 512 for i in range(1, 21)}  # example, adjust according to each dataset
    dataset = 'dataset_name'  # example dataset name
    max_epoch = 100
    loss_alpha1 = 1.0
    distill_alpha = 0.5
    PCL_scale = 1.0
    MPCL_alpha = 0.5
    aug_tau = 0.5
    distill = 'KL'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_data_path = os.path.join('dataset', 'part_one_dataset', 'train_data')
eval_data_path = os.path.join('dataset', 'part_one_dataset', 'eval_data')
train_path = os.path.join(base_data_path, '1_train_data.tar.pth')
eval_path = os.path.join(eval_data_path, '1_eval_data.tar.pth')
save_dir = "C://Users//ANUSHKA SINGH//CS771-Mini-Project-2//saved_models"
os.makedirs(save_dir, exist_ok=True)  # Create save directory if it doesn't exist



# Load the data (Assume data files D1 to D20 are loaded as separate datasets)
for i in range(1, 21):
    # Update dataset name and load data
    args = Args()
    args.dataset = f'D{i}'

    # Load data and targets (assuming a data loading function)
    # Replace `load_data` with actual loading logic
    data_path = os.path.join(base_data_path, f'{i}_train_data.tar.pth')  # Example path
    data = torch.load(data_path, weights_only=False)
    # Assuming `data` is a dictionary with 'images' and 'labels' for dataset i
    images = data['data']
    labels = data['targets']

    # Create the model and optimizer
    model = LWPWithRaTP(args).to(device)
    model.get_optimizer()

    # Training loop (simplified example)
    for epoch in range(args.max_epoch):
        # Here we simulate `minibatches` with your data for each epoch.
        minibatches = [images, labels]
        train_stats = model.train_source(minibatches, task_id=i, epoch=epoch)

        # Print progress
        if epoch % 10 == 0:
            print(f"Training D{i} - Epoch {epoch} - Loss: {train_stats['loss']}")

    # Save the trained model
    model_save_path = os.path.join(save_dir, f'LWP_D{i}.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f"Saved model for D{i} at {model_save_path}")
