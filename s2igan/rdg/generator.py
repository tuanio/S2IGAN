import torch
from torch import nn


class ConditioningAugmentationNetwork(nn.Module):
    def __init__(
        self,
        speech_emb_dim: int = 1024,
        gan_emb_dim: int = 128,
        negative_slope: float = 0.2,
    ):
        super().__init__()
        self.speech_dim = speech_emb_dim
        self.gan_dim = gan_emb_dim
        self.fc = nn.Linear(self.speech_dim, self.gan_dim * 4)
        # multiply by 2 if leaky relu
        # multiply by 4 if GLU
        # self.act = nn.LeakyReLU(negative_slope)
        self.act = nn.GLU()

    def forward(self, x):
        """
        x: speech embedding vector
        """
        x = self.act(self.fc(x))
        mu = x[:, : self.speech_dim]
        logvar = x[:, self.speech_dim :]
        std = torch.exp(log_var.mul(0.5))

        z = torch.randn(std.size()).to(x.device)
        z_ca = mu + std * z
        return z_ca, mu, log_var


class UpBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(
                in_channels,
                out_channels * 2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            n.BatchNorm2d(out_planes * 2),
            nn.GLU(),
        )

    def forward(self, x):
        return self.seq(x)


class InitStateGenerator(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int):
        super().__init__()
        # in_dim = z_dim + gan_dim
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.input_projection = nn.Sequential(
            nn.Linear(self.in_dim, self.hid_dim * 4 * 4 * 2),
            nn.BatchNorm1d(self.hid_dim * 4 * 4 * 2),
            nn.GLU(),
        )
        self.seq_upsample = nn.Sequential(
            UpBlock(self.hid_dim, self.hid_dim // 2),
            UpBlock(self.hid_dim // 2, self.hid_dim // 4),
            UpBlock(self.hid_dim // 4, self.hid_dim // 8),
            UpBlock(self.hid_dim // 8, self.hid_dim // 16),
        )

    def forward(self, z_code, c_code):
        inp = torch.cat((z_code, c_code), 1)
        out = self.input_projection(inp)
        out = out.view(-1, self.hid_dim, 4, 4)
        # up from 4x4 -> 64x64
        out = self.seq_upsample(out)
        return out


class ResBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels * 2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels * 2),
            GLU(),
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
        )

    def forward(self, x):
        return self.block(x) + x


class NextStageGenerator(nn.Module):
    def __init__(self, gan_emb_dim: int, hid_dim: int):
        super().__init__()
        self.gan_emb_dim = gan_emb_dim
        self.hid_dim = hid_dim
        self.joint_conv = nn.Sequential(
            nn.Conv2d(
                gan_emb_dim + hid_dim,
                hid_dim * 2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_planes * 2),
            GLU(),
        )
        self.residual = nn.Sequential(ResBlock(hid_dim), ResBlock(hid_dim))
        self.upsample = UpBlock(hid_dim * 2, hid_dim // 2)

    def forward(self, h_code, c_code):
        s_size = h_code.size(2)
        c_code = c_code.view(-1, self.gan_emb_dim, 1, 1)
        c_code = c_code.repeat(1, 1, s_size, s_size)

        concat_code = torch.cat((c_code, h_code), 1)

        out = self.joint_conv(concat_code)
        out = self.residual(out)
        out = torch.cat((out_code, h_code), 1)
        out = self.upsample(out)
        return out


class ImageGenerator(nn.Module):
    def __init__(self, hid_dim: int):
        super().__init__()
        self.img = nn.Sequential(
            nn.Conv2d(
                gan_emb_dim + hid_dim,
                hid_dim * 2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.Tanh(),
        )

    def forward(self, h_code):
        return self.img(h_code)


class DenselyStackedGenerator(nn.Module):
    def __init__(
        self, latent_space_dim: int, speech_emb_dim: int, gan_emb_dim: int, hid_dim: int
    ):
        super().__init__()
        inp_dim = latent_space_dim + speech_emb_dim

        self.conditioning_augmentation = ConditioningAugmentationNetwork(
            speech_emb_dim=speech_emb_dim, gan_emb_dim=gan_emb_dim
        )

        self.F0 = InitStateGenerator(in_dim=inp_dim, hid_dim=hid_dim * 16)
        self.G0 = ImageGenerator(hid_dim=hid_dim)

        self.F1 = NextStageGenerator(gan_emb_dim, hid_dim=hid_dim)
        self.G1 = ImageGenerator(hid_dim=hid_dim // 2)

        self.F2 = NextStageGenerator(gan_emb_dim, hid_dim=hid_dim // 2)
        self.G2 = ImageGenerator(hid_dim=hid_dim // 4)

    def forward(self, z_code, speech_emb):
        c_code, mu, logvar = self.conditioning_augmentation(speech_emb)

        h0 = self.F0(z_code, c_code)
        h1 = self.F1(h0, c_code)
        h2 = self.F2(h1, c_code)

        fake_imgs = [self.G0(h0), self.G1(h1), self.G2(h2)]

        return fake_imgs, mu, logvar
