"""
Arquitectura DCGAN (Radford et al., 2016) para imagenes 64x64 del universo Lumen.

Elegimos DCGAN como base porque es la referencia estandar para empezar simple:
convoluciones transpuestas en G, convoluciones con stride en D, BatchNorm y
activaciones ReLU/LeakyReLU. Sobre esta base aplicamos los experimentos.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm as sn


def _maybe_sn(module: nn.Module, use_sn: bool) -> nn.Module:
    """Envuelve una capa con normalizacion espectral si use_sn es True."""
    return sn(module) if use_sn else module


class Generator(nn.Module):
    """Mapea z (nz) -> imagen 3x64x64 en [-1, 1] mediante conv transpuestas."""

    def __init__(self, nz: int = 128, ngf: int = 64, channels: int = 3):
        super().__init__()
        self.nz = nz
        self.net = nn.Sequential(
            # z: (nz,1,1) -> (ngf*8,4,4)
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # -> (ngf*4,8,8)
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # -> (ngf*2,16,16)
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # -> (ngf,32,32)
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # -> (channels,64,64)
            nn.ConvTranspose2d(ngf, channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 2:
            z = z.view(z.size(0), z.size(1), 1, 1)
        return self.net(z)


class Discriminator(nn.Module):
    """Clasifica 3x64x64 -> logit. Soporta normalizacion espectral opcional."""

    def __init__(self, ndf: int = 64, channels: int = 3, use_sn: bool = False):
        super().__init__()
        # Sin BatchNorm cuando se usa normalizacion espectral (patron habitual de SNGAN).
        use_bn = not use_sn

        def block(cin, cout, bn):
            layers = [_maybe_sn(nn.Conv2d(cin, cout, 4, 2, 1, bias=not bn), use_sn)]
            if bn:
                layers.append(nn.BatchNorm2d(cout))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.net = nn.Sequential(
            # (channels,64,64) -> (ndf,32,32)  (primera capa sin BN por convencion)
            *block(channels, ndf, False),
            # -> (ndf*2,16,16)
            *block(ndf, ndf * 2, use_bn),
            # -> (ndf*4,8,8)
            *block(ndf * 2, ndf * 4, use_bn),
            # -> (ndf*8,4,4)
            *block(ndf * 4, ndf * 8, use_bn),
            # -> logit (1,1,1)
            _maybe_sn(nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False), use_sn),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1)


def weights_init(m: nn.Module) -> None:
    """Inicializacion DCGAN: Normal(0, 0.02) en conv y BN."""
    classname = m.__class__.__name__
    if "Conv" in classname:
        if m.weight is not None:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


def build_models(cfg, device):
    """Construye G y D segun la configuracion y los inicializa."""
    G = Generator(cfg.nz, cfg.ngf, cfg.channels).to(device)
    D = Discriminator(cfg.ndf, cfg.channels, use_sn=cfg.spectral_norm).to(device)
    G.apply(weights_init)
    D.apply(weights_init)
    return G, D
