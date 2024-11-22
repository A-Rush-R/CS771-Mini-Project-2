from typing import Optional
import torch

from ..distances.distance_base import Distance

class LearningWithPrototype:
    def __init__(self, num_classes: int, device: torch.device, distance_type: Distance, features: Optional[torch.tensor], labels: Optional[torch.tensor]) -> None:
        self.num_classes = num_classes
        self.device = device
        self.distance_type = distance_type
        self.features = features
        self.labels = labels
        # self.prototypes = [] if self.features is None else self.compute_prototypes()
    

    def compute_prototypes(self) -> torch.tensor:
        if self.features is None or self.labels is None:
            raise ValueError("Features and labels must be provided")
        prototypes = [torch.ones(self.features.shape[1]) for i in range(self.num_classes)]
        
        for i in range(self.num_classes):
            prototypes[i] = torch.mean(self.features[self.labels == i], dim=0)
        
        return prototypes
    
    def predict(self, features: torch.tensor) -> torch.tensor:
        prototypes = self.compute_prototypes()
        dd = self.distance_type.distance(features, torch.stack(prototypes))
        return torch.argmin(dd, dim=1)

    def update(self, features: torch.tensor, labels: torch.tensor) -> None:
        if features is None or labels is None:
            raise ValueError("Features and labels must be provided")
        self.features = torch.cat((self.features, features)) if self.features is not None else features
        self.labels = torch.cat((self.labels, labels)) if self.labels is not None else labels
        self.distance_type.update(self)
        

    def eval(self, test_features: torch.tensor, test_labels: torch.tensor) -> float:
        predictions = self.predict(test_features)
        accuracy = torch.sum(predictions == test_labels).item() / len(test_labels)
        return accuracy