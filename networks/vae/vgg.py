import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, opt):
        super().__init__()
        px = opt.width**2
        self.feat = nn.Sequential(
            nn.Conv2d(opt.image_chan, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(64, 64, 4, stride=1, padding=0),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(int(64 * px / 16), opt.z_dim)
        self.fc_logvar = nn.Linear(int(64 * px / 16), opt.z_dim)

    def forward(self, x):
        x = self.feat(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        z = torch.randn_like(mu)
        z = z * std + mu
        return z


class Decoder(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.z_dim = opt.z_dim
        self.width = opt.width
        self.px = self.width ** 2

        self.fc = nn.Sequential(
            nn.Linear(self.z_dim, int(64 * self.px / 16)),
            nn.ReLU()
        )
        self.model = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, opt.image_chan, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        batch_size = z.shape[0]
        t = self.fc(z).view(batch_size, -1,
                            int(self.width / 4),
                            int(self.width / 4))
        x = self.model(t)
        return x


class BasicVAE(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.encoder = Encoder(opt)
        self.decoder = Decoder(opt)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.encoder.reparameterize(mu, logvar)
        x = self.decoder(z)
        return mu, logvar, x
