import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis, flop_count_str
import torch.nn.functional as F


class WDCNN(nn.Module):
    """
    WDCNN builder
    """

    def __init__(self, ) -> object:
        super(WDCNN, self).__init__()
        self.cnn = nn.Sequential()
        self.cnn.add_module('Conv1D_1', nn.Conv1d(1, 16, 64, 8, 28))
        self.cnn.add_module('BN_1', nn.BatchNorm1d(16))
        self.cnn.add_module('Relu_1', nn.ReLU())
        self.cnn.add_module('MAXPool_1', nn.MaxPool1d(2, 2))
        self.__make_layer(16, 32, 1, 2)
        self.__make_layer(32, 64, 1, 3)   #  改64
        self.__make_layer(64, 64, 1, 4)   #  改64
        self.__make_layer(64, 64, 1, 5)   #  改64
        self.__make_layer(64, 64, 0, 6)
        self.fc1 = nn.Linear(192, 100)
        self.relu1 = nn.ReLU()
        self.dp = nn.Dropout(0.5)
        self.fc2 = nn.Linear(100, 10)

    def __make_layer(self, in_channels, out_channels, padding, nb_patch):
        self.cnn.add_module('Conv1D_%d' % (nb_patch), nn.Conv1d(in_channels, out_channels, 3, 1, padding))
        self.cnn.add_module('BN_%d' % (nb_patch), nn.BatchNorm1d(out_channels))
        self.cnn.add_module('ReLu_%d' % (nb_patch), nn.ReLU())
        self.cnn.add_module('MAXPool_%d' % (nb_patch), nn.MaxPool1d(2, 2))

    def forward(self, x):
        out = self.cnn(x)
        out = self.fc1(out.view(x.size(0), -1))
        out = self.relu1(out)
        out = self.dp(out)
        out = self.fc2(out)
        return F.softmax(out, dim=1)


if __name__ == '__main__':
    X = torch.rand(1, 1, 2048)
    m = WDCNN()
    print(flop_count_str(FlopCountAnalysis(m, X)))