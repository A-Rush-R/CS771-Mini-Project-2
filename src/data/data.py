from typing import List, Tuple
from torchvision import datasets, transforms, models
import torch

def get_raw_data() -> Tuple[List[torch.tensor]]:
    train_paths = ["dataset/part_one_dataset/train_data", "dataset/part_two_dataset/train_data"]
    test_paths = ["dataset/part_one_dataset/eval_data", "dataset/part_two_dataset/eval_data"]
    x_train, y_train, x_test, y_test = [], [], [], []
    for pth in train_paths:
        for i in range(1, 11):
            path = f"{pth}/{i}_train_data.tar.pth"
            t = torch.load(path, weights_only = False)
            # print(t.keys())
            # continue
            data = t['data']

            if i == 1 and pth == train_paths[0]:
                y_train.append(torch.tensor(t['targets']))
            x_tensor = torch.tensor(data, dtype = torch.float32).permute(0,3,1,2)
            x_train.append(x_tensor)
            
    for pth in test_paths:
        for i in range(1, 11):
            path = f"{pth}/{i}_eval_data.tar.pth"
            t = torch.load(path, weights_only = False)
            data, targets = t['data'], t['targets']

            x_tensor = torch.tensor(data, dtype = torch.float32).permute(0,3,1,2)
            y_tensor = torch.tensor(targets)

            
            x_test.append(x_tensor)
            y_test.append(y_tensor)
        
    return x_train, y_train, x_test, y_test

def resnet_embeddings() -> Tuple[List[torch.tensor]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        embeds  = torch.load('stuff/embeds.pt', map_location = device)
        return embeds
    except:
        x_train, y_train, x_test, y_test = get_raw_data()
        transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224 (ResNet input size)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
        ])

        resnet =  models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # Remove the last layer
        resnet.eval()  # Set to evaluation mode

        # Move model to GPU if available
        resnet = resnet.to(device)

    




