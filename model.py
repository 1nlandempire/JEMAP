# implementation of 2D-CNN model in pytorch
import torch
import torch.nn as nn
import math

class Res_Block(nn.Module):
    def __init__(self, C_in, C_out):
        super(Res_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=C_in, out_channels=C_out, kernel_size=1, padding=0)

        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=C_out, out_channels=C_out, kernel_size=3, padding=1)

        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=C_out, out_channels=C_out, kernel_size=3, padding=1)

        self.relu3 = nn.ReLU()

    def forward(self, x):
        y1 = self.conv1(x)
        y1 = self.relu1(y1)
        y2 = self.conv2(y1)
        y2 = self.relu2(y2)
        y2 = self.conv3(y2)
        y2 = self.relu3(y2)
        y = torch.add(y2, y1)
        return y

class Out_Block(nn.Module):
    def __init__(self, C_in, C_out):
        super(Out_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=C_in, out_channels=128, kernel_size=1, padding=0)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=C_out, kernel_size=1, padding=0)

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.conv2(y)
        return y

class Model(nn.Module):
    def __init__(self, n_channels, n_out=11, res_hiddens=[128, 256]):
        '''
        n_channels  ---> number of input DWIs
        n_out       ---> number of output diffusion parameters
        res_hiddens ---> number of hidden features in res blocks

        '''
        super(Model, self).__init__()
        layers = []
        layers.append(Res_Block(n_channels, res_hiddens[0]))
        for ii in range(1, len(res_hiddens)):
            layers.append(Res_Block(res_hiddens[ii - 1], res_hiddens[ii]))

        layers.append(Out_Block(res_hiddens[ii], n_out))

        self.JEMAP = nn.Sequential(*layers)

    def forward(self, x):
        out = self.JEMAP(x)
        return out


def main():

    num_channels = 3
    batch_size = 16
    JEMAP = Model(n_channels=num_channels, n_out=16)
    print(JEMAP)

    #
    input = torch.Tensor(torch.rand(batch_size, num_channels, 112, 112))
    output = JEMAP(input)
    print(f'Input size is {input.shape}:')
    print(f'Output size is {output.shape}')


if __name__ == "__main__":
    main()



