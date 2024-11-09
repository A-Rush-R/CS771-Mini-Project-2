import torch
import sklearn
from distance_base import Distance

class EuclideanDistance(Distance):

    def __init__(self, device) -> None:
        super().__init__()
        self.device = device

    def update(self) -> None:
        pass
    
    def distance(x: torch.tensor, y: torch.tensor) -> torch.tensor:
        x_np, y_np = x.numpy(), y.numpy()
        return torch.tensor(sklearn.metrics.pairwise.euclidean_distances(x_np, y_np)).to(self.device)
