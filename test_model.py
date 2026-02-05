import torch
import torch.nn as nn
import torch.nn.functional as F

class IICNet(nn.Module):
    def __init__(self, in_channels=3, num_clusters=10, image_size=32):
        super(IICNet, self).__init__()
        self.num_clusters = num_clusters
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.cluster_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_clusters),
        )
    def forward(self, x):
        features = self.encoder(x)
        logits = self.cluster_head(features)
        return F.softmax(logits, dim=1)

model = IICNet(num_clusters=10)
checkpoint = torch.load('models/iic_cifar10_model.pth', map_location='cpu', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
print('Modele charge avec succes!')
print('Accuracy:', checkpoint['final_accuracy']*100, '%')
print('NMI:', checkpoint['final_nmi'])
