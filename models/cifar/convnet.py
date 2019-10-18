#
# Simple convnet for experiments with few parameters.
#

import torch
import torch.nn as nn
import torch.nn.functional as F

#
#
#
class ConvNet(nn.Module):
    
    #
    #
    #
    def __init__(self, k1, k2, k3, k4, nr_outputs, polars=None):
        super(ConvNet, self).__init__()
        
        # Store a copy of the class prototypes.
        self.polars = polars
        
        # First three convs.
        self.conv11 = nn.Conv2d(3, k1, kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(k1, k1, kernel_size=3, padding=1)
        self.conv13 = nn.Conv2d(k1, k1, kernel_size=3, padding=1)
        
        # Second three convs.
        self.conv21 = nn.Conv2d(k1, k2, kernel_size=3, padding=1)
        self.conv22 = nn.Conv2d(k2, k2, kernel_size=3, padding=1)
        self.conv23 = nn.Conv2d(k2, k2, kernel_size=3, padding=1)
        
        # First three convs.
        self.conv31 = nn.Conv2d(k2, k3, kernel_size=3, padding=1)
        self.conv32 = nn.Conv2d(k3, k3, kernel_size=3, padding=1)
        self.conv33 = nn.Conv2d(k3, k3, kernel_size=3, padding=1)
        
        # FC.
        self.fcin = 4*4*k3
        self.fc1  = nn.Linear(self.fcin, k4)
        self.fc2  = nn.Linear(k4, nr_outputs)
    
    
    #
    #
    #
    def forward(self, x):
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = F.relu(self.conv12(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv21(x))
        x = F.relu(self.conv22(x))
        x = F.relu(self.conv22(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv31(x))
        x = F.relu(self.conv32(x))
        x = F.relu(self.conv32(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, self.fcin)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    #
    #
    #
    def predict(self, x):
        # Normalize the outputs (so that we can use dot product).
        x = F.normalize(x, p=2, dim=1)
        # Dot product with polar prototypes.
        x = torch.mm(x, self.polars.t().cuda())
        return x
