import torch.nn as nn
import torch.nn.functional as F

class FCClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, ndf=64, avg_pool=True, avg_pool_size=1):
        super(FCClassifier, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, ndf, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
        self.classifier = nn.Conv2d(ndf*8, num_classes, kernel_size=1, stride=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.avg_pool = avg_pool
        if self.avg_pool:
            self.reduce_dim = nn.AdaptiveAvgPool2d(avg_pool_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        if self.avg_pool:
            x = self.reduce_dim(x)
        x = self.classifier(x)
        
        return x


class FCClassifierBatchNorm(nn.Module):
    def __init__(self, input_dim, num_classes, ndf=64, avg_pool=True, avg_pool_size=1):
        super(FCClassifierBatchNorm, self).__init__()
        self.main = nn.Sequential(
            # input is (input_dim) x w x h
            nn.Conv2d(input_dim, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x w/2 x h/2
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x w/4 x h/4
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x w/8 x h/8
            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x w/16 x h/16
        )
        self.avg_pool = avg_pool
        if self.avg_pool:
            self.reduce_dim = nn.AdaptiveAvgPool2d(avg_pool_size)
        self.classifier = nn.Conv2d(ndf*8, num_classes, 1, 1, 0)

    def forward(self, x):
        x = self.main(x)
        if self.avg_pool:
            x = self.reduce_dim(x)
        x = self.classifier(x)
        return x