import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FeatureExtractor(nn.Module):
    """A simple feature extractor. Replace with a more complex model if needed."""
    def __init__(self, feature_dim):
        super(FeatureExtractor, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc = nn.Linear(128 * 8 * 8, feature_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class LinearLayerPrototypes(nn.Module):
    """Classifier with linear layer where weights act as class prototypes."""
    def __init__(self, feature_dim, num_classes):
        super(LinearLayerPrototypes, self).__init__()
        self.classifier = nn.Linear(feature_dim, num_classes, bias=False)  # Set bias to zero
        nn.init.kaiming_uniform_(self.classifier.weight, mode='fan_out', a=math.sqrt(5))

    def forward(self, x):
        return self.classifier(x)

    def get_prototypes(self):
        """Returns the classifier weights as prototypes."""
        return self.classifier.weight

class DomainAlignmentModel(nn.Module):
    """Main model combining feature extractor, classifier, and loss functions."""
    def __init__(self, feature_dim, num_classes, old_model=None):
        super(DomainAlignmentModel, self).__init__()
        self.feature_extractor = FeatureExtractor(feature_dim)
        self.classifier = LinearLayerPrototypes(feature_dim, num_classes)
        self.old_model = old_model

    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return features, logits

    def contrastive_alignment_loss(self, features, labels, prototypes):
        """
        Computes contrastive loss to align prototypes across domains.
        Positive pairs: features with their corresponding prototype.
        Negative pairs: features with prototypes of other classes.
        """
        batch_size = features.size(0)
        loss = 0.0

        for i in range(batch_size):
            positive_sim = F.cosine_similarity(features[i], prototypes[labels[i]], dim=0)
            negative_sims = [F.cosine_similarity(features[i], prototypes[j], dim=0)
                             for j in range(prototypes.size(0)) if j != labels[i]]
            negative_sim = torch.stack(negative_sims).mean()
            loss += F.relu(1 - positive_sim + negative_sim)

        return loss / batch_size

    def distillation_loss(self, new_logits, old_logits, temperature=2.0):
        """
        Knowledge distillation loss between new and old model outputs.
        """
        old_probs = F.softmax(old_logits / temperature, dim=1)
        new_log_probs = F.log_softmax(new_logits / temperature, dim=1)
        loss = F.kl_div(new_log_probs, old_probs, reduction="batchmean") * (temperature ** 2)
        return loss

    def compute_loss(self, features, logits, labels):
        """
        Compute total loss including CrossEntropy, Contrastive Alignment, and Distillation loss.
        """
        # CrossEntropy loss for classification
        ce_loss = F.cross_entropy(logits, labels)

        # Contrastive alignment loss with prototypes
        prototypes = self.classifier.get_prototypes()
        contrastive_loss = self.contrastive_alignment_loss(features, labels, prototypes)

        # Distillation loss if old model is provided
        if self.old_model is not None:
            with torch.no_grad():
             _, old_logits = self.old_model(features)
            distill_loss = self.distillation_loss(logits, old_logits)
        else:
            distill_loss = torch.tensor(0.0).to(features.device)

        # Total loss with weighting factors
        total_loss = ce_loss + 0.5 * contrastive_loss + 0.5 * distill_loss
        return total_loss, {'ce_loss': ce_loss.item(), 'contrastive_loss': contrastive_loss.item(), 'distill_loss': distill_loss.item()}

# Example usage
feature_dim = 128
num_classes = 10
model = DomainAlignmentModel(feature_dim, num_classes).to(device)
old_model = DomainAlignmentModel(feature_dim, num_classes).to(device)  # Initialize or load the old model

# Dummy data and labels
data = torch.randn(32, 3, 32, 32).to(device)
labels = torch.randint(0, num_classes, (32,)).to(device)

# Forward pass
features, logits = model(data)
loss, loss_dict = model.compute_loss(features, logits, labels)

print("Total loss:", loss.item())
print("Loss components:", loss_dict)
