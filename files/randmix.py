import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the Autoencoder with convolutional layers
class SimpleAutoencoder(nn.Module):
    def __init__(self):
        super(SimpleAutoencoder, self).__init__()
        self.encoder = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.decoder = nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Define AdaIN for noise injection
class AdaIN(nn.Module):
    def __init__(self):
        super(AdaIN, self).__init__()
        self.l1 = nn.Linear(1, 1)
        self.l2 = nn.Linear(1, 1)

    def forward(self, x):
        n = torch.randn(1)  # Normalized noise
        noise_1 = self.l1(n)
        noise_2 = self.l2(n)
        return noise_1 * x + noise_2

# Random Mixup function with multiple autoencoders
class RandMix:
    def __init__(self, num_autoencoders=5):
        self.autoencoders = [SimpleAutoencoder() for _ in range(num_autoencoders)]
        self.adain = AdaIN()

    def mixup(self, x):
        augmented_data = []
        for ae in self.autoencoders:
            encoded = ae.encoder(x)
            noisy_encoded = self.adain(encoded)
            decoded = ae.decoder(noisy_encoded)
            augmented_data.append(decoded)

        # Combine outputs with random weights
        weights = torch.softmax(torch.randn(len(self.autoencoders)), dim=0)
        mix = weights[0] * x  # Start with the original data scaled by weight
        for i in range(1, len(self.autoencoders)):
            mix += weights[i] * augmented_data[i]

        return torch.sigmoid(mix)

    def augment_batch(self, batch):
        augmented_batch = []
        for x in batch:
            augmented_batch.append(self.mixup(x))
        return torch.stack(augmented_batch)

# Initialize RandMix and augment a batch of data
randmix = RandMix(num_autoencoders=5)
batch_data = torch.randn(10, 3, 32, 32)  # Example batch with 10 images (3 channels, 32x32)
augmented_batch = randmix.augment_batch(batch_data)
