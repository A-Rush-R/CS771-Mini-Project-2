from typing import Optional
import torch

from distances.distance_base import Distance

class LearningWithPrototype:
    def __init__(self, num_classes: int, device: torch.device, distance_type: Distance, features: Optional[torch.tensor], labels: Optional[torch.tensor]) -> None:
        self.num_classes = num_classes
        self.device = device
        self.distance_type = distance_type
        self.features = features
        self.labels = labels
    

    def compute_prototypes(self, features: torch.tensor, labels: torch.tensor) -> torch.tensor:
        if features is None or labels is None:
            raise ValueError("Features and labels must be provided")
        prototypes = torch.zeros((self.num_classes, features.shape[1])).to(self.device)
        for i in range(self.num_classes):
            prototypes[i] = torch.mean(features[labels == i], dim=0)
        
        return prototypes
    
    def predict(self, features: torch.tensor) -> torch.tensor:
        if self.prototypes is None:
            raise ValueError("Prototypes must be computed before predicting")
        distance = self.distance_type.distance(self, features, self.prototypes)
        return torch.argmin(distance, dim=1)

    def update(self, features: torch.tensor, labels: torch.tensor) -> None:
        if features is None or labels is None:
            raise ValueError("Features and labels must be provided")
        self.features = torch.cat((self.features, features)) if self.features is not None else features
        self.labels = torch.cat((self.labels, labels)) if self.labels is not None else labels
        self.distance_type.update(self)
        self.prototypes = self.compute_prototypes(self.features, self.labels)
        

    def eval(self, test_features: torch.tensor, test_labels: torch.tensor) -> float:
        if self.prototypes is None:
            raise ValueError("Prototypes must be computed before evaluating")
        predictions = self.predict(test_features)
        accuracy = torch.sum(predictions == test_labels).item() / len(test_labels)
        return accuracy