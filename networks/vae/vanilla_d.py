import torch
from torch import nn
from torch.nn import functional as F


class Encoder(nn.Module):
    def __init__(self, opt):
        super().__init__()
        in_channels = opt.image_chan
        self.latent_dim = opt.z_dim

        modules = []
        hidden_dims = opt.get('hidden_dims', [32, 64, 128, 256, 512])

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels,
                              out_channels = h_dim,
                              kernel_size = 3,
                              stride = 2, 
                              padding = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, self.latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4, self.latent_dim)

    def forward(self, x):
        x = self.encoder(x)

        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu


class Decoder(nn.Module):
    def __init__(self, opt):
        super().__init__()
        latent_dim = opt.z_dim
        hidden_dims = opt.get('hidden_dims', [32, 64, 128, 256, 512])

        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)
        hidden_dims.reverse()
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size = 3,
                                       stride = 2,
                                       padding = 1,
                                       output_padding = 1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[-1],
                                        hidden_dims[-1],
                                        kernel_size = 3,
                                        stride = 2,
                                        padding = 1,
                                        output_padding = 1),
                    nn.BatchNorm2d(hidden_dims[-1]),
                    nn.LeakyReLU(),
                    nn.Conv2d(hidden_dims[-1],
                                out_channels = 3,
                                kernel_size = 3,
                                padding = 1),
                    nn.Tanh())

    def forward(self, z):
        x = self.decoder_input(z)
        x = x.view(-1, 512, 2, 2)
        x = self.decoder(x)
        x = self.final_layer(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator).__init__()
        in_channels = opt.image_chan

        modules = []
        hidden_dims = opt.get('hidden_dims', [32, 64, 128, 256, 512])

        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels,
                              out_channels = h_dim,
                              kernel_size = 3,
                              stride = 2, 
                              padding = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.feat = nn.Sequential(*modules)
        self.gap = nn.AdaptiveAvgPool1d((1, 1))
        self.fc = nn.Linear(512, 1)

    def forward(self, x):
        x = self.feat(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class VanillaVAED(nn.Module):
    def __init__(self, opt):
        super(VanillaVAED, self).__init__()
        self.encoder = Encoder(opt)
        self.decoder = Decoder(opt)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.encoder.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return mu, logvar, x_hat

    def sample(self, num_samples, device, **kwargs):
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(device)

        samples = self.decode(z)
        return samples
