import torch
import torch.nn.functional as F
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader

class Top2PseudoLabeling:
    def __init__(self, model, num_classes=10, top_k_ratio=0.5):
        """
        Initialize Top2PseudoLabeling with the given model and configuration.
        
        Args:
            model: A pretrained model with a feature extractor and classifier.
            num_classes: Number of classes.
            top_k_ratio: Ratio of top samples to use for centroid construction.
        """
        self.model = model
        self.num_classes = num_classes
        self.top_k_ratio = top_k_ratio

    def generate_pseudo_labels(self, dataloader):
        """
        Generates pseudo labels for a given dataloader using Top-2 Pseudo Labeling (T2PL).
        
        Args:
            dataloader: DataLoader for the dataset to be labeled.
        
        Returns:
            pseudo_labels: Assigned pseudo-labels for each sample in the dataset.
        """
        # Set model to evaluation mode and initialize lists for storing features and outputs
        self.model.eval()
        all_features, all_outputs, all_labels = [], [], []

        # Forward pass to get features and outputs
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.cuda()
                features = self.model.featurizer(images)  # Assuming featurizer extracts features
                outputs = F.softmax(self.model(features), dim=1)  # Classifier outputs softmax scores
                
                all_features.append(features.cpu())
                all_outputs.append(outputs.cpu())
                all_labels.append(labels)

        # Concatenate all batches
        all_features = torch.cat(all_features, dim=0)
        all_outputs = torch.cat(all_outputs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # Initialize pseudo labels tensor
        pseudo_labels = torch.zeros(all_features.size(0), dtype=torch.long)
        
        # Perform Top-2 Pseudo Labeling for each class
        for cls in range(self.num_classes):
            # Mask for class and select top-K% based on confidence
            class_mask = all_labels == cls
            class_features = all_features[class_mask]
            class_outputs = all_outputs[class_mask, cls]  # Confidence for current class
            top_k = max(int(len(class_outputs) * self.top_k_ratio), 1)
            
            # Get top K% indices by confidence
            _, top_indices = torch.topk(class_outputs, top_k)
            top_features = class_features[top_indices]
            
            # Construct centroid for the current class
            centroid = torch.mean(top_features, dim=0)
            
            # Calculate cosine similarity with all samples and select the top set
            cos_sim = F.cosine_similarity(all_features, centroid.unsqueeze(0), dim=1)
            sorted_sim, sorted_indices = torch.sort(cos_sim, descending=True)
            selected_indices = sorted_indices[:top_k]

            # Fit kNN on the selected features
            knn = KNeighborsClassifier(n_neighbors=5)  # Using k=5 neighbors
            knn.fit(top_features.numpy(), [cls] * top_features.size(0))
            
            # Assign pseudo-labels for top samples in each class based on kNN
            pseudo_labels[selected_indices] = torch.tensor(knn.predict(all_features[selected_indices].numpy()))

        return pseudo_labels
