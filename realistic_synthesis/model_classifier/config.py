class Config:
  def __init__(self, num_conv, num_lin, output, kernel, stride, padding, lin_value, maxpool, dim):
    self.num_conv = num_conv
    self.num_lin = num_lin
    self.output = output
    self.kernel = kernel
    self.stride = stride
    self.padding = padding
    self.lin_value = lin_value
    self.maxpool = maxpool
    self.dim = dim


num_conv = 4
num_lin = 2
output = [1,8, 16, 32]
kernel = [3,3,3,3]
padding = [1,1,1,1]
stride = [1,1,1,1]
lin_value = [3136, 512, 10]
maxpool = [False, True, False, True]
dim = 7

# configuration for Minst classifier (version 1) the difference is nn.sequential and therefore different values for lsa
# so basically in nn.sequential when we do layerwise activation we use all that its icluded relu, linear, dropout or something 
# else while in v2 its only the layer itself
mConfig = Config(num_conv, num_lin, output, kernel, stride, padding, lin_value, maxpool, dim)

num_conv = 4
num_lin = 2
output = [1,32,32 ,64,64]
kernel = [3,3,3,3]
padding = [1,1,1,1]
stride = [1,1,1,1]
lin_value = [3136, 512, 10]
maxpool = [False, True, False, True]
dim = 7

# configuration for Minst classifier (version 2)
mConfig1 = Config(num_conv, num_lin, output, kernel, stride, padding, lin_value, maxpool, dim)