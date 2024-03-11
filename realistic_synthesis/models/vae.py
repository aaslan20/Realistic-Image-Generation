import torch
import torch.nn as nn
import torch.nn.functional as F



# base Autoencoder (only for testing purpose)
class Autoencoder(nn.Module):
    def __init__(self, encoded_space_dim, fc2_input_dim):
        super(Autoencoder, self).__init__()

        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True)
        )

        self.flatten = nn.Flatten(start_dim=1)

        self.encoder_lin = nn.Sequential(
            nn.Linear(3 * 3 * 32, fc2_input_dim),
            nn.ReLU(True),
            nn.Linear(fc2_input_dim, encoded_space_dim)
        )

        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, fc2_input_dim),
            nn.ReLU(True),
            nn.Linear(fc2_input_dim, 3 * 3 * 32),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 3, 3))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        return x
    

# Dynamic Conv Vae (the one which will get used we have the MINST variant already hard coded)
class DCVAE(nn.Module):
  def __init__(self, config):
    super(DCVAE, self).__init__()

    enc_layers = []
    dec_layers = []

    for i in range(config.num_layers):
      enc_layers.append(nn.Conv2d(config.output[i], config.output[i + 1], config.kernel[i],
                                        config.stride[i], config.padding[i]))
      enc_layers.append(nn.RELU(True))
      # cause batchnorm is not used every time!
      if config.batch[i]:
        enc_layers.append(nn.BatchNorm2d(config.output[i]))

      self.encoder_cnn = nn.Sequential(*enc_layers)

      self.flatten = nn.Flatten(start_dim=1)

      self.encoder_lin_mean = nn.Linear(config.f_map * config.f_map * config.output[-i], config.encoded_space_dim)
      self.encoder_lin_logvar = nn.Linear(config.f_map * config.f_map * config.output[-i], config.encoded_space_dim)

      self.decoder_lin = nn.Sequential(
            nn.Linear(config.encoded_space_dim, config.fc2_input_dim),
            nn.ReLU(True),
            nn.Linear(config.fc2_input_dim, config.f_map * config.f_map * config.output[-i]),
            nn.ReLU(True)
        )

      self.unflatten = nn.Unflatten(dim=1, unflattened_size=(config.f_map, config.f_map, config.output[-i]))

      config.output =  config.output[::-1]
      config.output =  config.output[:-1][::-1]
      for i in range(config.num_layers - 1):
        dec_layers.append(nn.Conv2d(config.output[i], config.output[i + 1], config.d_kernel[i], 
                                        config.d_stride[i], config.d_padding[i], config.d_output_padding[i]))
        dec_layers.append(nn.ReLU(True))
        dec_layers.append(nn.BatchNorm2d(config.output[i + 1]))

      dec_layers.append(nn.Conv2d(config.output[-1], config.output[-1], config.d_kernel[-1], 
                                        config.d_stride[-1], config.d_padding[-1], config.d_output_padding[-1]))
      dec_layers.append(nn.Sigmoid())
      
      self.decoder_cnn = nn.Sequential(*dec_layers)

  def encoder(self, x):
    x = self.encoder_cnn(x)
    x = self.flatten(x)
    mean = self.encoder_lin_mean(x)
    logvar = self.encoder_lin_logvar(x)
    return mean, logvar

  def reparameterization(self, mean, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    z = mean + eps * std
    return z

  def decoder(self, z):      
    x = self.decoder_lin(z)
    x = self.unflatten(x)
    x = self.decoder_cnn(x)
    return x

  def forward(self, x):
    mean, logvar = self.encoder(x)
    latent = self.reparameterization(mean, logvar)
    reconstruction = self.decoder(latent)
    return reconstruction, mean, logvar 

  def loss_function(self, recon_x, x, mu, logvar):
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    loss = recon_loss + kl_divergence
    return loss

# Conv VAE (will be used for MINST)
class ConvVAE(nn.Module):
    def __init__(self, encoded_space_dim, fc2_input_dim):
        super(ConvVAE, self).__init__()
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True)
        )

        self.flatten = nn.Flatten(start_dim=1)

        self.encoder_lin_mean = nn.Linear(3 * 3 * 32, encoded_space_dim)
        self.encoder_lin_logvar = nn.Linear(3 * 3 * 32, encoded_space_dim)

        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, fc2_input_dim),
            nn.ReLU(True),
            nn.Linear(fc2_input_dim, 3 * 3 * 32),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 3, 3))

        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def encoder(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        mean = self.encoder_lin_mean(x)
        logvar = self.encoder_lin_logvar(x)
        return mean, logvar

    def reparameterization(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z

    def decoder(self, z):      
        x = self.decoder_lin(z)
        x = self.unflatten(x)
        x = self.decoder_cnn(x)
        return x

    def forward(self, x):
        mean, logvar = self.encoder(x)
        latent = self.reparameterization(mean, logvar)
        reconstruction = self.decoder(latent)
        return reconstruction, mean, logvar 

    def loss_function(self, recon_x, x, mu, logvar):
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + kl_divergence
        return loss


# dynamic Residual
class ResidualBlock2(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers, kernel_size, padding, stride, skip = True):
        super(ResidualBlock, self).__init__()
        self.skip = skip

        layers = []

        for _ in range(num_layers):
            layers.append(nn.Conv2d(in_dim, out_dim, kernel_size= 3, padding= 1, stride= stride))
            layers.append(nn.BatchNorm2d(out_dim))
            if _ < num_layers:
                layers.append(nn.LeakyReLU(0.2))

       
        if skip is True:
            self.conv = nn.Sequential(
            nn.Conv2d(out_dim, in_dim , kernel_size= 1, stride= stride),
            nn.BatchNorm2d(in_dim),
            )

        self.network = nn.Sequential(*layers)
        if self.skip:
            residual = self.conv3(x)
        out += residual

        return nn.LeakyReLU()(out)

# hard coded residual
class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, skip = True):
        super(ResidualBlock, self).__init__()
        self.skip = skip
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size= 3, padding= 1, stride= 1),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_dim,out_dim, kernel_size= 3, padding= 1, stride= 1),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(0.2)
        )

        if skip is True:
            self.conv3 = nn.Sequential(
            nn.Conv2d(out_dim, in_dim , kernel_size= 1, stride= 1),
            nn.BatchNorm2d(in_dim),
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.skip:
            residual = self.conv3(x)
        out += residual

        return nn.LeakyReLU()(out)
    
def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.skip:
            residual = self.conv3(x)
        out += residual

        return nn.LeakyReLU()(out)

# Conv VAE dynamic with residual (like sinvad code but dynamic)    
class ConvVAEResidual(nn.Module):
    def __init__(self, config, latent_dim, hidden_dim, max_pool):
        super(ConvVAEResidual, self).__init__()
        self.config = config

        enc_layers = []
        dec_layers = []

        # output lists need last element double!
        for i in range(config.num_layers):
            enc_layers.append(nn.Conv2d(config.output[i], config.output[i + 1], config.e_kernel[i], 
                                        config.e_stride[i], config.e_padding[i]))
            enc_layers.append(nn.ReLU(True))
            enc_layers.append(nn.BatchNorm2d(config.output[i + 1]))
            enc_layers.append(ResidualBlock(config.output[i + 1], config.output[i + 1]))
            enc_layers.append(ResidualBlock(config.output[i + 1], config.output[i + 1]))
            enc_layers.append(nn.MaxPool2d(max_pool))

        # convolutional encoder
        self.encoder_cnn = nn.Sequential(*enc_layers)

        # flatten
        self.flatten = nn.Flatten(start_dim=1)
        
        # linear encoder
        self.encoder_lin = nn.Sequential(
            nn.Linear(config.output[0] * config.output[0] * config.output[-1], latent_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # linear decoder
        self.decoder_lin = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, config.output[0] * config.output[0] * config.output[-1]),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(config.output[-1], config.output[0] , config.output[0]))


        
        # remove last item  & reserve list
        config.output =  config.output[:-1][::-1]
        dec_layers.append(ResidualBlock(config.output[0], config.output[0]))
        dec_layers.append(ResidualBlock(config.output[0], config.output[0]))
        for i in range(config.num_layers - 1):
            dec_layers.append(nn.Conv2d(config.output[i], config.output[i + 1], config.d_kernel[i], 
                                        config.d_stride[i], config.d_padding[i], config.d_output_padding[i]))
            dec_layers.append(nn.ReLU(True))
            dec_layers.append(nn.BatchNorm2d(config.output[i + 1]))

        dec_layers.append(nn.Conv2d(config.output[-1], config.output[-1], config.d_kernel[-1], 
                                        config.d_stride[-1], config.d_padding[-1], config.d_output_padding[-1]))
        dec_layers.append(nn.Sigmoid())

        # convolutional decoder
        self.decoder_cnn = nn.Sequential(*dec_layers)


    def encode(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        mu = self.encoder_lin(x)
        log_var = self.encoder_lin(x)
        return mu, log_var

    def decode(self, z):
        out = self.decoder_lin(z)
        out = self.unflatten(out)
        out = self.decoder_conv(out)
        return out

    def reparameterize(self, mu, log_var):
        sigma = torch.exp(log_var/2)
        return mu + sigma * torch.randn(sigma.size()).device

    def forward(self, x):
        mu, log_var = self.encode(x)
        latent_z = self.reparameterize(mu, log_var)
        reconstruction = self.decode(latent_z)
        return reconstruction, mu, log_var
    

    




""" class VAE(nn.Module):
    def __init__(self, config, max_pool):
        super(VAE, self).__init__()
        self.config = config
        self.max_pool = max_pool


        enc_layers = []
        dec_layers = []


        for i in range(config.num_layers):
            enc_layers.append(nn.Conv2d(config.output[i], config.output[i + 1], config.e_kernel[i], 
                                        config.e_stride[i], config.e_padding[i]))
            enc_layers.append(nn.ReLU(True))
            enc_layers.append(nn.BatchNorm2d(config.output[i + 1]))
            enc_layers.append(ResidualBlock(config.output[i + 1], config.output[i + 1]))
            enc_layers.append(ResidualBlock(config.output[i + 1], config.output[i + 1]))
            enc_layers.append(nn.MaxPool2d(max_pool))

        # convolutional encoder
        self.encoder_cnn = nn.Sequential(*enc_layers)

        # flatten
        self.flatten = nn.Flatten(start_dim=1)
        
        # linear encoder
        self.encoder_lin = nn.Sequential(
            nn.Linear(config.output[0] * config.output[0] * config.output[-1], config.fc2_input_dim),
            nn.ReLU(True),
            nn.Linear(config.fc2_input_dim, config.latent)
        )
        
        # linear decoder
        self.decoder_lin = nn.Sequential(
            nn.Linear(config.fc2_input_dimm, config.latent),
            nn.ReLU(True),
            nn.Linear(config.fc2_input_dim, config.num_c * config.num_c * config.output[-1]),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(config.output[-1], config.output[0] , config.output[0]))


        
        # reserve list
        config.output =  config.output[::-1]
        dec_layers.append(ResidualBlock(config.output[0], config.output[0]))
        dec_layers.append(ResidualBlock(config.output[0], config.output[0]))
        for i in range(config.num_layers):
            dec_layers.append(nn.Conv2d(config.output[i], config.output[i + 1], config.d_kernel[i], 
                                        config.d_stride[i], config.d_padding[i], config.d_output_padding[i]))
            enc_layers.append(nn.ReLU(True))
            enc_layers.append(nn.BatchNorm2d(config.output[i + 1]))

            if i == config.num_layers -1:
                dec_layers.append(nn.Conv2d(config.output[i], config.output[i + 1], config.d_kernel[i], 
                                        config.d_stride[i], config.d_padding[i], config.d_output_padding[i]))
                enc_layers.append(nn.Sigmoid(True))
                break

        # convolutional decoder
        self.decoder_cnn = nn.Sequential(*dec_layers)


    def encode(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        mu = self.encoder_lin(x)
        log_var = self.encoder_lin(x)
        return mu, log_var

    def decode(self, z):
        out = self.decoder_lin(z)
        out = self.unflatten(out)
        out = self.decoder_conv(out)
        return out

    def reparameterize(self, mu, log_var):
        sigma = torch.exp(log_var/2)
        return mu + sigma * torch.randn(sigma.size()).cuda()

    def forward(self, x):
        mu, log_var = self.encode(x)
        latent_z = self.reparameterize(mu, log_var)
        reconstruction = self.decode(latent_z)
        return reconstruction, mu, log_var
    def forward(self, x):
        out = self.network(x)




        




        








# for rgb dataset
class Autoencoder(nn.Module):
    def __init__(self, encoded_space_dim, fc2_input_dim):
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(32),

            ResidualBlock(32, 32, skip = True),
            ResidualBlock(32, 32, skip = True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            ResidualBlock(64, 64, skip = True),
            ResidualBlock(64, 64, skip = True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(128),

            ResidualBlock(128, 128, skip = True),
            ResidualBlock(128, 128, skip = True),
            nn.MaxPool2d(2),
        )

        self.flatten = nn.Flatten(start_dim=1)

        self.encoder_lin = nn.Sequential(
            nn.Linear(3 * 3 * 128, fc2_input_dim),
            nn.ReLU(True),
            nn.Linear(fc2_input_dim, encoded_space_dim)
        )

        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, fc2_input_dim),
            nn.ReLU(True),
            nn.Linear(fc2_input_dim, 3 * 3 * 128),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(128, 3, 3))

        self.decoder_conv = nn.Sequential(
            ResidualBlock(128, 128, skip = True),
            ResidualBlock(128, 128, skip = True),


            nn.ConvTranspose2d(128, 64, 3, stride=2,  output_padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(True),


            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=0, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 1, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        mu = self.encoder_lin(x)
        log_var = self.encoder_lin(x)
        return mu, log_var

    def decode(self, z):
        out = self.decoder_lin(z)
        out = self.unflatten(out)
        out = self.decoder_conv(out)
        return out

    def reparameterize(self, mu, log_var):
        sigma = torch.exp(log_var/2)
        return mu + sigma * torch.randn(sigma.size()).cuda()

    def forward(self, x):
        mu, log_var = self.encode(x)
        latent_z = self.reparameterize(mu, log_var)
        reconstruction = self.decode(latent_z)
        return reconstruction, mu, log_var """
    









""" 
# dynamic
class VAE2(nn.Module):
    def __ini__(self, config):
        super(VAE2, self).__init__()
        self.cofig = config

        enc_layers = []
        dec_layers = []

        enc_layers.append(nn.Conv2d(config.num_c, config.out, config.kernel, config.padding))
        enc_layers.append(nn.ReLU(True))
        enc_layers.append(nn.BatchNorm2d(config.out))
        enc_layers.append(ResidualBlock(config.output, config.output))
        enc_layers.append(ResidualBlock(config.output, config.output))
        enc_layers.append(nn.MaxPool2d(2))

        for _ in range(config.num_layers):
            enc_layers.append(nn.Conv2d(config.output, config.output * 2, kernel_size=config.kernel, stride=config.stride, padding=config.padding, bias=False))
            enc_layers.append(nn.BatchNorm2d(config.output * 2))
            enc_layers.append(nn.LeakyReLU(0.2, inplace=True))
            enc_layers.append(ResidualBlock(config.output, config.output))
            enc_layers.append(ResidualBlock(config.output, config.output))
            enc_layers.append(nn.MaxPool2d(2))
            config.output *= 2
    
        # has the manual and dynamically created layers for encoding
        self.encoder_cnn = nn.Sequential(*enc_layers)

        # flatten
        self.flatten = nn.Flatten(start_dim=1)

        # calculation of the last output in enc_layers
        last_output = config.output * config.numl_layers * 2

        # linear encoder
        self.encoder_lin = nn.Sequential(
            nn.Linear(config.num_c * config.num_c * last_output, config.fc2_input_dim),
            nn.ReLU(True),
            nn.Linear(config.fc2_input_dim, config.latent)
        )

        # linear decoder
        self.decoder_lin = nn.Sequential(
            nn.Linear(config.latent, config.fc2_input_dim),
            nn.ReLU(True),
            nn.Linear(config.fc2_input_dim, config.num_c * config.num_c * last_output),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(last_output, config.num_c, config.num_c))



        dec_layers.append(ResidualBlock(last_output, last_output)),
        dec_layers.append(ResidualBlock(last_output, last_output))
        for _ in range(config.num_layers):
            dec_layers.append(nn.ConvTranspose2d(last_output, (last_output // 2), 3, stride=2,  output_padding=0))
            dec_layers.append(nn.BatchNorm2d())
            dec_layers.append(nn.ReLU(True))
            last_output = last_output // 2
            if _ == config.num_layers - 1:
                dec_layers.append(last_output, config.num_c, stride = 2, padding = 1, output_padding = 1)
                dec_layers.append(nn.Sigmoid)
                break;
        
        self.decoder_cnn = nn.Sequential(*dec_layers)

    def encode(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        mu = self.encoder_lin(x)
        log_var = self.encoder_lin(x)
        return mu, log_var

    def decode(self, z):
        out = self.decoder_lin(z)
        out = self.unflatten(out)
        out = self.decoder_conv(out)
        return out

    def reparameterize(self, mu, log_var):
        sigma = torch.exp(log_var/2)
        return mu + sigma * torch.randn(sigma.size()).cuda()

    def forward(self, x):
        mu, log_var = self.encode(x)
        latent_z = self.reparameterize(mu, log_var)
        reconstruction = self.decode(latent_z)
        return reconstruction, mu, log_var """