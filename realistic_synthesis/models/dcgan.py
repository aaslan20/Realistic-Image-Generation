import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

def weights_init(m):
  classname = m.__class__.__name__
  if classname.find("Conv") != -1:
    nn.init.normal_(m.weight.data, 0.0, 0.02)
  elif classname.find('BatchNorm') != -1:
    nn.init.normal_(m.weight.data, 1.0, 0.02)
    nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()

        layers = []

        for i in range(config.num_layers):
            layers.append(nn.ConvTranspose2d(config.output[i], config.output[i + 1], config.kernel[i],
                                        config.stride[i], config.padding[i], bias=False))
            layers.append(nn.BatchNorm2d(config.output[i + 1]))
            layers.append(nn.ReLU(True))

        layers.append(nn.ConvTranspose2d(config.output[-2], config.output[-1], config.kernel[-1],
                                        config.stride[-1], config.padding[-1], bias=False))
        layers.append(nn.Tanh())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
      return self.network(x)

class Discriminator(nn.Module):
    def __init__(self, config):
       super(Discriminator, self).__init__()
       layers = []

       for i in range(config.num_layers):
           layers.append(nn.Conv2d(config.output[i], config.output[i + 1], config.kernel[i],
                                        config.stride[i], config.padding[i]))
           layers.append(nn.BatchNorm2d(config.output[i + 1]))
           layers.append(nn.ReLU(True))

       layers.append(nn.Conv2d(config.output[-2], config.output[-1], config.kernel[-1],
                                        config.stride[-1], config.padding[-1]))
       layers.append(nn.Sigmoid())
       self.network = nn.Sequential(*layers)

    def forward(self, x):
      return self.network(x)
    
class CGenerator(nn.Module):
    def __init__(self, config):
        super(CGenerator, self).__init__()
        self.embedding = nn.Embedding(config.num_classes, config.embedding)
        layers = []
        self.input = config.output[0] 
        
        for i in range(config.num_layers):
            if i == 0:
                input_dim = config.output[i] + config.embedding
            else:
                input_dim = config.output[i]

            layers.append(nn.ConvTranspose2d(input_dim, config.output[i + 1], config.kernel[i],
                                        config.stride[i], config.padding[i], bias=False))
            layers.append(nn.BatchNorm2d(config.output[i + 1]))
            layers.append(nn.ReLU(True))

        layers.append(nn.ConvTranspose2d(config.output[-2], config.output[-1], config.kernel[-1],
                                        config.stride[-1], config.padding[-1], bias=False))
        layers.append(nn.Tanh())

        self.network = nn.Sequential(*layers)

    def forward(self, x, labels):
      embedded_labels = self.embedding(labels).unsqueeze(2).unsqueeze(3)
      embedded_labels = embedded_labels.expand(x.size(0), -1, x.size(2), x.size(3))
      x = torch.cat((x, embedded_labels), dim=1)
      return self.network(x)

class CDiscriminator(nn.Module):
  def __init__(self, config):
    super(CDiscriminator, self).__init__()
    self.embedding = nn.Embedding(config.num_classes, config.embedding)
    self.input = config.output[0] 
    layers = []
    for i in range(config.num_layers):
      if i == 0:
        input_dim = config.output[i] + config.embedding
      else:
        input_dim = config.output[i]
      layers.append(nn.Conv2d(input_dim, config.output[i + 1], config.kernel[i],
                                        config.stride[i], config.padding[i]))
      layers.append(nn.BatchNorm2d(config.output[i + 1]))
      layers.append(nn.ReLU(True))

    layers.append(nn.Conv2d(config.output[-2], config.output[-1], config.kernel[-1],
                                        config.stride[-1], config.padding[-1]))
    layers.append(nn.Sigmoid())
    self.network = nn.Sequential(*layers)

  def forward(self, x, labels):
    embedded_labels = self.embedding(labels).unsqueeze(2).unsqueeze(3)
    embedded_labels = embedded_labels.expand(x.size(0), -1, x.size(2), x.size(3))
    x = torch.cat((x, embedded_labels), dim=1)
    x = self.network(x)
    return x.squeeze()
  

    

# blocks to create a model example generator = nn.Sequential(gen_blocks(1,64,3,1, bn = False)
                                                            #gen_blocks(64,128,3,1, bn = True)
#                                                                 )

def dis_blocks(input, output, kernel, stride, padding, bn = True, activation = False):
  layers = [nn.Conv2d(input, output, kernel_size= kernel, stride= stride, padding= padding, bias=False)]
  if bn: layers.append(nn.BatchNorm2d(output))
  layers.append(nn.Sigmoid() if activation else nn.LeakyReLU(0.2, inplace=True))
  return nn.Sequential(*layers)


def gen_blocks(input, output, kernel, stride, padding, bn = True, activation = False):
  layers = [nn.Conv2d(input, output, kernel_size= kernel, stride= stride, padding= padding, bias=False)]
  if bn: layers.append(nn.BatchNorm2d(output))
  layers.append(nn.Tanh() if activation else nn.ReLU(True))
  return nn.Sequential(*layers)