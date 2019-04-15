import torch
import torch.nn as nn
import torch.nn.functional as F
from models.BasicModule import BasicModule
class DCD(BasicModule):
    def __init__(self,h_features=64,input_features=128):
        super(DCD,self).__init__()

        self.fc1=nn.Linear(input_features,h_features)
        self.fc2=nn.Linear(h_features,h_features)
        self.fc3=nn.Linear(h_features,4)

    def forward(self,inputs):
        out=F.relu(self.fc1(inputs))
        out=self.fc2(out)
        return F.softmax(self.fc3(out),dim=1)

class Classifier(BasicModule):
    def __init__(self,opt):
        super(Classifier,self).__init__()
        self.fc=nn.Linear(opt['classifier_input_dim'],23)

    def forward(self,input):
        return F.softmax(self.fc(input),dim=1)


class Encoder(BasicModule):
    def __init__(self, opt):
        super(Encoder,self).__init__()
        hid_dim = opt['encoder_hid_dim']
        z_dim = opt['encoder_z_dim']
        self.encoder = nn.Sequential(
            conv_block(1, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
            Flatten()
        )


    def forward(self,input):
        return self.encoder(input)


def conv_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


