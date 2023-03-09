import torch
import torch.nn as nn
import config

class Yolov1(nn.Module):
    def __init__(self, in_channels = 3):
        super().__init__()

        # Defining the depth
        self.depth = config.B * 5 + config.C

        layers = [
            # Conv1
            nn.Conv2d(in_channels, 64, kernel_size=7, stride = 2, padding = 3),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size = 2, stride= 2),

            # Conv2
            nn.Conv2d(64, 192, kernel_size=3, padding = 1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size = 2, stride= 2),

            # Conv3
            nn.Conv2d(192, 128, kernel_size=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, kernel_size=3, padding = 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256,512, kernel_size=3, padding = 1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size = 2, stride= 2)
        ]

        # Conv4
        for _ in range(4):
            layers +=[
                nn.Conv2d(512, 256, kernel_size=1),
                nn.LeakyReLU(0.1),
                nn.Conv2d(256,512, kernel_size=3, padding = 1),
                nn.LeakyReLU(0.1)
                ]
        layers+=[
            nn.Conv2d(512, 512, kernel_size=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512,1024, kernel_size=3, padding = 1),
            nn.MaxPool2d(kernel_size = 2, stride =2)
            ]

        # Conv5
        for _ in range(2):
            layers+=[
                nn.Conv2d(1024, 512, kernel_size=1),
                nn.LeakyReLU(0.1),
                nn.Conv2d(512,1024, kernel_size=3, padding = 1),
                nn.LeakyReLU(0.1)
            ]
        layers+= [
            nn.Conv2d(1024, 1024,kernel_size=3, padding = 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024,kernel_size=3, stride = 2, padding = 1)
        ]

        # Conv6
        layers +=[
            nn.Conv2d(1024,1024,3,padding = 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024,1024,3,padding = 1),
            nn.LeakyReLU(0.1)
        ]

        # Block 7
        layers += [
            nn.Flatten(),
            nn.Linear(config.S * config.S * 1024, 4096),
            nn.Dropout(),
            nn.LeakyReLU(0.1),
            nn.Linear(4096,config.S * config.S * self.depth),
        ]

        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
         return torch.reshape(
            self.model.forward(x),
            (x.size(dim=0), config.S, config.S, self.depth)
        )

def test():
    model  = Yolov1()
    x = torch.randn((2,3,448,448))
    print(model(x).shape)

test()
