import torch
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from torchvision import models


class PrototypeBasedModelWithKMeans:
    def __init__(self, train_path, eval_path, num_prototypes=5):
        self.train_path = train_path
        self.eval_path = eval_path
        self.num_prototypes = num_prototypes  # Number of prototypes (clusters)
        self.model = self.load_model()
    
    def load_model(self):
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # Remove the last layer
        resnet.eval() 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        resnet = resnet.to(device)
        
        return resnet
    
    def load_data(self, file_path):
        t = torch.load(file_path, weights_only=False)
        features = t['data']
        # If features is a Tensor, convert to numpy array
        if isinstance(features, torch.Tensor):
            features_np = features.cpu().numpy() if features.is_cuda else features.numpy()
        else:
            features_np = features  # Assume features is already a NumPy array
        return features_np
    
    # Function to apply k-means clustering to the features and calculate weights
    def apply_kmeans_and_calculate_weights(self, features):
        print("Original shape of features:", features.shape)
        # Reshape features for clustering (flatten each image's feature vector)
        features_reshaped = features.reshape(features.shape[0], -1)
        print("Reshaped features shape:", features_reshaped.shape)
        
        # Fit k-means with reshaped features
        kmeans = KMeans(n_clusters=self.num_prototypes, random_state=42)
        kmeans.fit(features_reshaped)
        
        # The cluster centers are your initial prototypes
        prototypes = kmeans.cluster_centers_
        
        # Calculate distances of each point to its cluster center
        distances = np.linalg.norm(features_reshaped - kmeans.cluster_centers_[kmeans.labels_], axis=1)
        
        # Convert distances to weights (closer points have higher weights)
        weights = 1 / (distances + 1e-8)  # Add a small constant to avoid division by zero
        weights /= weights.max()  # Normalize to [0, 1]
        
        return prototypes, weights

    def evaluate(self):
        features_train = self.load_data(self.train_path)
        
        # Apply k-means and calculate weights for the training data
        prototypes, weights = self.apply_kmeans_and_calculate_weights(features_train)
        
        print(f"Prototypes (cluster centers):\n{prototypes[:5]}")  # Displaying first 5 prototypes for inspection
        print(f"Sample weights:\n{weights[:10]}")  # Displaying first 10 weights for inspection

        # Load the evaluation data and apply the same process
        features_eval = self.load_data(self.eval_path)
        
        # Apply k-means and calculate weights for the evaluation data
        prototypes_eval, weights_eval = self.apply_kmeans_and_calculate_weights(features_eval)
        
        print(f"Prototypes for evaluation data:\n{prototypes_eval[:5]}")
        print(f"Sample weights for evaluation data:\n{weights_eval[:10]}")


if __name__ == "__main__":
    train_dir = os.path.join('dataset', 'part_one_dataset', 'train_data')
    eval_dir = os.path.join('dataset', 'part_one_dataset', 'eval_data')
    
    train_path = os.path.join(train_dir, '1_train_data.tar.pth')
    eval_path = os.path.join(eval_dir, '1_eval_data.tar.pth')
    
    # Instantiate the PrototypeBasedModelWithKMeans class
    prototype_model = PrototypeBasedModelWithKMeans(train_path=train_path, eval_path=eval_path, num_prototypes=5)
    
    # Evaluate the model on training and evaluation data
    prototype_model.evaluate()
    