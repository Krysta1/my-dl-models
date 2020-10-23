import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()

        self.feature = nn.Sequential(
            nn.Conv2d(3, 96, 11, 4, 2),
            # output h = (h + 2 * padding - kernel_size) / stride + 1
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(96, 256, 5, 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(256, 384, 3, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, 3, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3, 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2)
        )
        # get a output size you wanted.
        # If the input is N * C * H * W. And you parameter is (6, 6)
        # output size is N * C * 6 * 6, to keep different input sizes get the same output sizes.
        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 10),
        )

    def forward(self, x):
        x = self.feature(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
