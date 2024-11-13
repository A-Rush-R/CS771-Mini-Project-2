import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision import models, transforms
import numpy as np
from RandMix import RandMix, AdaIN2d  # Import RandMix and AdaIN classes
from PCA import PCLoss, PCALoss
from torchvision.models import ResNet50_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResNetFeatureExtractor(nn.Module):
    def __init__(self, fea_dim):
        super(ResNetFeatureExtractor, self).__init__()
        # Load pretrained ResNet model and remove the fully connected layer
        resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.resnet_layers = nn.Sequential(*list(resnet.children())[:-1])  # Exclude the final FC layer
        self.fc = nn.Linear(resnet.fc.in_features, fea_dim)  # Adjust to output desired feature dimensions

    def forward(self, x):
        x = self.resnet_layers(x)
        x = torch.flatten(x, 1)  # Flatten the output of the ResNet conv layers
        x = self.fc(x)  # Map to the desired feature dimension
        return x
class LWPWithRaTP(nn.Module):
    def __init__(self, args):
        super(LWPWithRaTP, self).__init__()
        self.args = args
        self.task_id = 0
        fea_dim = args.proj_dim[args.dataset]
        
        self.featurizer = ResNetFeatureExtractor(fea_dim)
        self.encoder = nn.Sequential(
            nn.Linear(fea_dim, fea_dim),
            nn.ReLU(),
            nn.Linear(fea_dim, fea_dim)
        )
        
        # Classifier
        self.classifier = nn.Parameter(torch.FloatTensor(args.num_classes, fea_dim))
        nn.init.kaiming_uniform_(self.classifier, mode='fan_out', a=np.sqrt(5))
        
        # Initialize RandMix for data augmentation
        self.data_aug = RandMix(noise_lv=1).to(device)
        if args.dataset == 'dg5':
            self.aug_tran = transforms.Normalize([0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        else:
            self.aug_tran = transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, x):
        # Forward pass with feature extraction and classification
        x = self.featurizer(x)
        x = self.encoder(x)
        self.fea_rep = x
        pred = F.linear(x, self.classifier)
        return pred

    def get_optimizer(self, lr_decay=1.0):
        # Define optimizer with weight decay and learning rate adjustments
        self.optimizer = torch.optim.SGD([
            {'params': self.featurizer.parameters(), 'lr': lr_decay * self.args.lr},
            {'params': self.encoder.parameters()},
            {'params': self.classifier},
        ], lr=self.args.lr, weight_decay=self.args.weight_decay)

    def train_source(self, minibatches, task_id, epoch):
        self.task_id = task_id
        all_x = torch.tensor(minibatches[0]).to(device).float()
        all_y = torch.tensor(minibatches[1]).to(device).long()

        # Apply RandMix augmentation and concatenate original and augmented data
        ratio = epoch / self.args.max_epoch
        data_fore = self.aug_tran(torch.sigmoid(self.data_aug(all_x, ratio=ratio)))
        if data_fore.shape[1] != 3:
            # Convert to 3 channels if necessary (e.g., using a channel reduction technique)
            # For example, you can take the first 3 channels or convert it to grayscale:
            data_fore = data_fore[:, :3, :, :]  # Keep only the first 3 channels

        if all_x.shape[1:] != data_fore.shape[1:]:  # Ensure compatibility for concatenation
            print(f"Shape mismatch: all_x has shape {all_x.shape}, data_fore has shape {data_fore.shape}")
            # Adjust `data_fore` shape if necessary to match `all_x`
            data_fore = data_fore.permute(0, 2, 3, 1) if data_fore.shape[1] != all_x.shape[1] else data_fore
        
        all_x = torch.cat([all_x, data_fore], dim=0)
        all_y = torch.cat([all_y, all_y], dim=0)

        # Compute PCA loss and backpropagate
        loss, loss_dict = self.PCAupdate(all_x, all_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'loss': loss.item(), **loss_dict}

    def adapt(self, minibatches, task_id, epoch, replay_dataloader=None, old_model=None):
        self.task_id = task_id
        all_x = torch.tensor(minibatches[0]).to(device).float()
        all_y = torch.tensor(minibatches[1]).to(device).long()

        # Apply RandMix augmentation during adaptation
        all_x, all_y = self.select_aug(all_x, all_y, epoch)
        loss, loss_dict = self.PCAupdate(all_x, all_y, old_model)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'loss': loss.item(), **loss_dict}

    def PCAupdate(self, all_x, all_y, old_model=None):
        # Calculate predictions
        pred = self(all_x)

        # Compute cross-entropy loss
        loss_cls = F.nll_loss(F.log_softmax(pred, dim=1), all_y)

        # Compute PCA loss with current and, if applicable, previous proxy
        proxy = self.classifier
        features = self.fea_rep
        if self.task_id > 0 and old_model is not None:
            old_proxy = old_model.classifier
            loss_pcl = PCALoss(self.args.num_classes, self.args.PCL_scale)(features, all_y, proxy, old_proxy, mweight=self.args.MPCL_alpha)
        else:
            loss_pcl = PCLoss(num_classes=self.args.num_classes, scale=self.args.PCL_scale)(features, all_y, proxy)

        # Combine losses
        loss = loss_cls + self.args.loss_alpha1 * loss_pcl
        loss_dict = {'ce_loss': loss_cls.item(), 'pcl_loss': loss_pcl.item()}

        # Optional: add distillation loss if an old model is provided
        if old_model is not None:
            distill_loss = self.args.distill_alpha * self.distill_loss(pred, all_x, old_model)
            loss += distill_loss
            loss_dict['distill_loss'] = distill_loss.item()

        return loss, loss_dict

    def distill_loss(self, pred, all_x, old_model):
        # Compute distillation loss with previous model outputs
        old_model.to(device).eval()
        with torch.no_grad():
            old_logits = F.softmax(old_model(all_x), dim=1)

        if self.args.distill == 'CE':
            loss = F.cross_entropy(pred, old_logits)
        elif self.args.distill == 'KL':
            loss = nn.KLDivLoss(reduction="batchmean")(F.log_softmax(pred, dim=1), old_logits)
        elif self.args.distill == 'feaKL':
            loss = nn.KLDivLoss(reduction="batchmean")(F.log_softmax(self.fea_rep, dim=1), F.softmax(old_model.fea_rep, dim=1))
        return loss

    def select_aug(self, all_x, all_y, epoch):
        ratio = epoch / self.args.max_epoch
        if self.args.aug_tau > 0:
            self.eval()
            with torch.no_grad():
                pred = F.softmax(self(all_x), dim=1)
                ov, idx = torch.max(pred, 1)
                bool_index = ov > self.args.aug_tau
                data_fore = all_x[bool_index]
                y_fore = all_y[bool_index]
                data_fore = self.aug_tran(torch.sigmoid(self.data_aug(data_fore, ratio=ratio)))
            self.train()
        else:
            data_fore = self.aug_tran(torch.sigmoid(self.data_aug(all_x, ratio=ratio)))
            y_fore = all_y

        all_x = torch.cat([all_x, data_fore])
        all_y = torch.cat([all_y, y_fore])
        return all_x, all_y
    
    
    
