"""
Funciones de perdida para la GAN.

Incluye las variantes que comparamos en los experimentos:
 - "bce"     : perdida no saturante (BCE con logits), la base de DCGAN.
 - "hinge"   : perdida hinge (Lim & Ye, 2017), comun en SNGAN.
 - "wgan-gp" : critica de Wasserstein con penalizacion de gradiente.

Cada funcion devuelve (loss_D, loss_G_fn) de forma homogenea para el loop.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# BCE no saturante
# --------------------------------------------------------------------------- #
def d_loss_bce(d_real, d_fake, label_smoothing: float = 0.0):
    """Perdida del discriminador con BCE. label_smoothing resta a la etiqueta real."""
    real_target = torch.full_like(d_real, 1.0 - label_smoothing)
    fake_target = torch.zeros_like(d_fake)
    loss_real = F.binary_cross_entropy_with_logits(d_real, real_target)
    loss_fake = F.binary_cross_entropy_with_logits(d_fake, fake_target)
    return loss_real + loss_fake


def g_loss_bce(d_fake):
    """Perdida NO saturante del generador: maximiza log D(G(z))."""
    target = torch.ones_like(d_fake)
    return F.binary_cross_entropy_with_logits(d_fake, target)


# --------------------------------------------------------------------------- #
# Hinge
# --------------------------------------------------------------------------- #
def d_loss_hinge(d_real, d_fake):
    return F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean()


def g_loss_hinge(d_fake):
    return -d_fake.mean()


# --------------------------------------------------------------------------- #
# WGAN-GP
# --------------------------------------------------------------------------- #
def d_loss_wgan(d_real, d_fake):
    return d_fake.mean() - d_real.mean()


def g_loss_wgan(d_fake):
    return -d_fake.mean()


def gradient_penalty(D, real, fake, device):
    """Penalizacion de gradiente de WGAN-GP sobre interpolaciones reales/falsas."""
    bs = real.size(0)
    eps = torch.rand(bs, 1, 1, 1, device=device)
    inter = (eps * real + (1 - eps) * fake).requires_grad_(True)
    d_inter = D(inter)
    grads = torch.autograd.grad(
        outputs=d_inter, inputs=inter,
        grad_outputs=torch.ones_like(d_inter),
        create_graph=True, retain_graph=True, only_inputs=True,
    )[0]
    grads = grads.view(bs, -1)
    return ((grads.norm(2, dim=1) - 1.0) ** 2).mean()
