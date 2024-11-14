from lwp import *
from distances.euclidean import EuclideanDistance

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device  = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.mps.is_available() else "cpu"))

features = torch.randn(10, 10).to(device)

euclid = EuclideanDistance(device=device)
model = LearningWithPrototype(num_classes=10, device=device, distance_type=euclid, features=features, labels=None)