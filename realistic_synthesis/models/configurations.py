# config for GAN
class GConfig:
  def __init__(self,num_layers, output, kernel, stride, padding):
    self.num_layers = num_layers
    self.output = output
    self.kernel = kernel
    self.stride = stride
    self.padding = padding

num_layers = 3
g_output = [100,128,64,32,1]
g_kernel = [4,3,4,4]
g_stride = [1,2,2,2]
g_padding = [0,1,1,1]


d_output = [1,32,64,128,1]
d_kernel = [4,4,3,4]
d_stride = [2,2,2,2]
d_padding = [1,1,1,0]


g_config = GConfig(num_layers,g_output, g_kernel, g_stride, g_padding)
d_config = GConfig(num_layers,d_output, d_kernel, d_stride, d_padding)

# config for DCGAN
class DCGonfig:
  def __init__(self,num_layers, output, kernel, stride, padding, num_classes, embedding):
    self.num_layers = num_layers
    self.output = output
    self.kernel = kernel
    self.stride = stride
    self.padding = padding
    self.num_classes = num_classes
    self.embedding = embedding


num_layers = 3
num_classes = 10
embedding = 10

g_output = [100,128,64,32,1]
g_kernel = [4,3,4,4]
g_stride = [1,2,2,2]
g_padding = [0,1,1,1]


d_output = [1,32,64,128,1]
d_kernel = [4,4,3,4]
d_stride = [2,2,2,2]
d_padding = [1,1,1,0]


cg_config = DCGonfig(num_layers,g_output, g_kernel, g_stride, g_padding, num_classes, embedding)
cd_config = DCGonfig(num_layers,d_output, d_kernel, d_stride, d_padding, num_classes, embedding)


# config for vae set it right
class VConfig:
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

num_conv = 3
encoded_dim = 5

output = [1,8 ,16, 32, 64]
kernel = [3,3,3, 3]
padding = [1,1,1,1]
stride = [1,1,1,1]
lin_value = [3136, 512, 10]
maxpool = [False, True, False, True]
dim = 7

# configuration for Minst
#mConfig = Config(num_conv, num_lin, output, kernel, stride, padding, lin_value, maxpool, dim)