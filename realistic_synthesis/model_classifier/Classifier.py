import torch
import torch.nn as nn
import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self, config):
        super(Classifier, self).__init__()

        conv_layers = []
        linear_layers = []

        for i in range(config.num_conv):
            conv_layers.append(nn.Conv2d(config.output[i], config.output[i + 1], config.kernel[i],
                                        config.stride[i], config.padding[i]))
            conv_layers.append(nn.ReLU())
            if config.maxpool[i]:
                conv_layers.append(nn.MaxPool2d(2))

        for i in range(config.num_lin):
            linear_layers.append(nn.Dropout)
            linear_layers.append(nn.Linear(config.lin_value[i], config.lin_value[i+1]))
            if i <= config.num_lin -1:
                linear_layers.append(nn.ReLU)

        self.conv = nn.Sequential(*conv_layers)
        self.linear = nn.Sequential(*linear_layers)

        # for the view in forward take last output and the dim after the convs (calculate it with w' = [w-k + 2*P/s] + 1)
        self.value = config.output[-1]
        self.dim = config.dim

    def at_layer(self, x, layer_idx = 0):
        out = x
        idx_offset = 0

        for idx, module in enumerate(self.conv):
            out = module(out)
            if idx == layer_idx + idx_offset:
                if isinstance(module, nn.Conv2d):
                    out = torch.mean(out, dim=[2, 3])
                return out

        idx_offset += len(self.conv)

        out = out.view(-1, self.value * self.dim * self.dim)  

        for idx, module in enumerate(self.linear):
            out = module(out)
            if idx == layer_idx + idx_offset:
                return out
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, self.value * self.dim * self.dim)
        x = self.linear(x)
        

