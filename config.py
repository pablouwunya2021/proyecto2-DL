"""
Configuracion central del Proyecto 2 - GAN de personajes "Lumen".

Todas las decisiones reproducibles (semillas, hiperparametros, rutas) viven aqui
para que los experimentos de perdida y estabilizacion cambien UNA sola cosa a la
vez respecto a esta configuracion base.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field

import numpy as np
import torch

# --------------------------------------------------------------------------- #
# Rutas del proyecto
# --------------------------------------------------------------------------- #
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "datos")
IMG_DIR = os.path.join(DATA_DIR, "lumen")           # dataset de entrenamiento
SAMPLE_DIR = os.path.join(DATA_DIR, "muestras")     # muestras del dataset para el informe
CKPT_DIR = os.path.join(ROOT, "checkpoints")
EVID_DIR = os.path.join(ROOT, "evidencia")
GALLERY_DIR = os.path.join(ROOT, "galeria")

for _d in (DATA_DIR, IMG_DIR, SAMPLE_DIR, CKPT_DIR, EVID_DIR, GALLERY_DIR):
    os.makedirs(_d, exist_ok=True)

# --------------------------------------------------------------------------- #
# Semillas (declaradas en el README)
# --------------------------------------------------------------------------- #
SEED_DATA = 20260101    # generacion procedural del dataset
SEED_TRAIN = 42         # inicializacion de pesos y orden de los lotes
SEED_EVAL = 7           # ruido z fijo para evaluar y comparar experimentos


def set_seed(seed: int) -> None:
    """Fija las semillas de random, numpy y torch para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Selecciona GPU (CUDA o Apple MPS) si existe, si no CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# Hiperparametros de la configuracion BASE
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    # datos
    image_size: int = 64
    channels: int = 3
    dataset_size: int = 5000

    # arquitectura
    nz: int = 128          # dimension del vector de ruido z
    ngf: int = 64          # ancho base del generador
    ndf: int = 64          # ancho base del discriminador

    # entrenamiento
    batch_size: int = 128
    epochs: int = 60
    lr_g: float = 2e-4
    lr_d: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999

    # variantes de experimento (se sobreescriben, nunca dos a la vez)
    loss: str = "bce"            # {"bce", "hinge", "wgan-gp"}
    spectral_norm: bool = False  # normalizacion espectral en D
    label_smoothing: float = 0.0 # suavizado de etiqueta real (0 = desactivado)
    gp_lambda: float = 10.0      # peso del gradient penalty (solo wgan-gp)
    n_critic: int = 1            # pasos de D por paso de G

    # logging / evidencia
    fixed_noise_n: int = 16      # rejilla de ruido fijo
    sample_every: int = 5        # guardar rejilla cada N epocas
    name: str = "base"

    def describe(self) -> str:
        return (
            f"{self.name} | loss={self.loss} sn={self.spectral_norm} "
            f"ls={self.label_smoothing} epochs={self.epochs} bs={self.batch_size}"
        )


def base_config() -> Config:
    """Configuracion base compartida por todos los experimentos."""
    return Config(name="base", loss="bce", spectral_norm=False)
