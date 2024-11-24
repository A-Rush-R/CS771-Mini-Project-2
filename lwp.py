import numpy as np
from sklearn.metrics.pairwise import cosine_distances, manhattan_distances

class LWP:
    """Learning Vector Prototypes with configurable distance function"""
    
    DISTANCE_FUNCTIONS = {
        'euclidean': lambda x, y: np.linalg.norm(x - y),
        'cosine': lambda x, y: cosine_distances(x.reshape(1, -1), y.reshape(1, -1))[0][0],
        'manhattan': lambda x, y: manhattan_distances(x.reshape(1, -1), y.reshape(1, -1))[0][0],
        'minkowski': lambda x, y, p=2: np.power(np.sum(np.power(np.abs(x - y), p)), 1/p)
    }
    
    def __init__(self, distance_metric='euclidean', **distance_params):
        """
            distance_params (dict): Additional parameters for the distance function
        """
        self.prototypes = {}
        self.class_counts = {i: 0 for i in range(10)}
        
        if callable(distance_metric):
            self.distance_fn = distance_metric
        elif distance_metric in self.DISTANCE_FUNCTIONS:
            if distance_metric == 'minkowski':
                p = distance_params.get('p', 2)
                self.distance_fn = lambda x, y: self.DISTANCE_FUNCTIONS[distance_metric](x, y, p)
            else:
                self.distance_fn = self.DISTANCE_FUNCTIONS[distance_metric]
        else:
            raise ValueError(f"Unknown distance metric: {distance_metric}. " 
                           f"Available metrics: {list(self.DISTANCE_FUNCTIONS.keys())}")

    def fit(self, features, labels):
        unique_labels = np.unique(labels)
        for label in unique_labels:
            samples = features[labels == label]
            num_samples = len(samples)
            
            if label not in self.prototypes:  # Original condition was: if label not in self.prototypes
                self.prototypes[label] = samples.mean(axis=0)
                self.class_counts[label] = len(samples)
            else:
                self.class_counts[label] += len(samples)
                self.prototypes[label] = (
                    (self.class_counts[label] - num_samples) / self.class_counts[label] * self.prototypes[label] +
                    num_samples / self.class_counts[label] * samples.mean(axis=0)
                )

    def predict(self, features):
        preds = []
        for feature in features:
            distances = {
                label: self.distance_fn(feature, proto)
                for label, proto in self.prototypes.items()
            }
            preds.append(min(distances, key=distances.get))
        return np.array(preds)