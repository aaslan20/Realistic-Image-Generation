import torch
import torch.nn as nn
import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self, config):
        super(Classifier, self).__init__()

        conv_layers = []
        linear_layers = []

        for i in range(config.num_conv):
            conv_layers.append(nn.Sequential(nn.Conv2d(config.output[i], config.output[i + 1], config.kernel[i],
                                        config.stride[i], config.padding[i]), 
                                        nn.ReLU()))
            if config.maxpool[i]:
                conv_layers.append(nn.MaxPool2d(2))

        for i in range(config.num_lin - 1):
            linear_layers.append(nn.Sequential(nn.Dropout(), 
                                               nn.Linear(config.lin_value[i], config.lin_value[i+1]), 
                                               nn.ReLU()))
            
        linear_layers.append(nn.Sequential(nn.Dropout(), nn.Linear(config.lin_value[-2], config.lin_value[-1])))

        self.conv = nn.Sequential(*conv_layers)
        self.linear = nn.Sequential(*linear_layers)

        # for the view in forward take last output and the dim after the convs (calculate it with w' = [w-k + 2*P/s] + 1)
        self.value = config.output[-1]
        self.dim = config.dim

    def at_by_layer(self, x, layer_idx = 0):
        current_idx = 0
        out = x
        for lyr in self.conv:
            out = lyr(out)
            if current_idx == layer_idx:
                at = torch.mean(out, dim=3)
                at = torch.mean(at, dim=2)
                return at
            else:
                current_idx += 1
        out = out.view(-1, 64*7*7)
        if current_idx == layer_idx:
            return out
        else:
            current_idx += 1
        for lyr in self.linear:
            out = lyr(out)
            if current_idx == layer_idx:
                return out
            else:
                current_idx += 1
        # should never reach here
        raise ValueError('layer_idx value %d failed match with layers' % layer_idx)
         
    def at_by_layer1(self, x, layer_idx):
        num_conv_layers = len(self.conv)
        num_linear_layers = len(self.linear)
        out = x
        if layer_idx < num_conv_layers:
            layer = self.conv[layer_idx]
            out = layer(x)  #
            at = torch.mean(out, dim=[2, 3])
            return at
        elif layer_idx < num_conv_layers + num_linear_layers:
            out = out.view(-1, self.value * self.dim * self.dim)
            linear_idx = layer_idx - num_conv_layers
            layer = self.linear[linear_idx]
            print(layer)
            return layer(out)  
        else:
            raise ValueError('layer_idx value %d failed match with layers' % layer_idx)
        
    def at_by_layer2(self, x, layer_idx):
        out = x
        idx_offset = 0

        for idx, module in enumerate(self.conv):
            out = module(out)
            if idx == layer_idx:
                if isinstance(module, nn.Conv2d):
                    out = torch.mean(out, dim=[2, 3])
                return out
        idx_offset += len(self.conv)

        out = out.view(-1, self.value * self.dim * self.dim)  

        for idx, module in enumerate(self.linear):
            out = module(out)
            if layer_idx == idx + idx_offset:
                return out
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, self.value * self.dim * self.dim)
        x = self.linear(x)
        return x
        

class MinstC(nn.Module):
    def __init__(self, config):
        super(MinstC, self).__init__()

        conv_layers = []
        linear_layers = []

        for i in range(config.num_conv):
            conv_layers.append(nn.Conv2d(config.output[i], config.output[i + 1], config.kernel[i],
                                        config.stride[i], config.padding[i]))
            conv_layers.append(nn.ReLU())
            if config.maxpool[i]:
                conv_layers.append(nn.MaxPool2d(2))

        for i in range(config.num_lin):
            linear_layers.append(nn.Dropout())
            linear_layers.append(nn.Linear(config.lin_value[i], config.lin_value[i+1]))
            if i <= config.num_lin -1:
                linear_layers.append(nn.ReLU())

        self.conv = nn.Sequential(*conv_layers)
        self.linear = nn.Sequential(*linear_layers)

        # for the view in forward take last output and the dim after the convs (calculate it with w' = [w-k + 2*P/s] + 1)
        self.value = config.output[-1]
        self.dim = config.dim

    def at_by_layer(self, x, layer_idx = 0):
        current_idx = 0
        out = x
        for lyr in self.conv:
            out = lyr(out)
            if current_idx == layer_idx:
                at = torch.mean(out, dim=3)
                at = torch.mean(at, dim=2)
                return at
            else:
                current_idx += 1
        out = out.view(-1, 64*7*7)
        if current_idx == layer_idx:
            return out
        else:
            current_idx += 1
        for lyr in self.linear:
            out = lyr(out)
            if current_idx == layer_idx:
                return out
            else:
                current_idx += 1
        # should never reach here
        raise ValueError('layer_idx value %d failed match with layers' % layer_idx)
 
    def at_by_layer1(self, x, layer_idx):
        num_conv_layers = len(self.conv)
        num_linear_layers = len(self.linear)
        out = x
        if layer_idx < num_conv_layers:
            layer = self.conv[layer_idx]
            out = layer(x)  #
            at = torch.mean(out, dim=[2, 3])
            return at
        out = out.view(-1, self.value * self.dim * self.dim)
        if layer_idx < num_conv_layers + num_linear_layers:
            linear_idx = layer_idx - num_conv_layers
            layer = self.linear[linear_idx]
            print(layer)
            return layer(out)  
        else:
            raise ValueError('layer_idx value %d failed match with layers' % layer_idx)
        
    def at_by_layer2(self, x, layer_idx):
        out = x
        idx_offset = 0

        for idx, module in enumerate(self.conv):
            out = module(out)
            if idx == layer_idx:
                if isinstance(module, nn.Conv2d):
                    out = torch.mean(out, dim=[2, 3])
                return out
        idx_offset += len(self.conv)

        out = out.view(-1, self.value * self.dim * self.dim)  

        for idx, module in enumerate(self.linear):
            out = module(out)
            if layer_idx == idx + idx_offset:
                return out

    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, self.value * self.dim * self.dim)
        x = self.linear(x)
        return x

