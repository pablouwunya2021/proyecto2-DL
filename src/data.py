"""
Universo "Lumen": generacion procedural del dataset y carga para entrenamiento.

El universo son espiritus guardianes elementales bioluminiscentes: siluetas
simetricas y centradas sobre fondo oscuro, con un nucleo brillante y una paleta
de color ligada a su elemento. Las imagenes de entrenamiento se generan con este
script (origen y licencia claros: creacion propia, dominio publico / CC0), de
modo que el dataset es totalmente reproducible a partir de una semilla.

Nota importante: estas imagenes son el MATERIAL DE ENTRENAMIENTO. La galeria final
la produce el modelo GAN entrenado, no este generador procedural.
"""
from __future__ import annotations

import os
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torch
from torch.utils.data import Dataset, DataLoader

# --------------------------------------------------------------------------- #
# Definicion del universo: elementos y paletas
# --------------------------------------------------------------------------- #
# Cada elemento define un rango de tono (HSV) coherente para mantener el estilo.
ELEMENTOS = {
    "brasa":  {"hue": (0.00, 0.08)},   # rojo-naranja (fuego)
    "marea":  {"hue": (0.52, 0.62)},   # azul-cian (agua)
    "musgo":  {"hue": (0.28, 0.38)},   # verde (tierra/bosque)
    "cefiro": {"hue": (0.45, 0.52)},   # verde-azulado palido (aire)
    "vacio":  {"hue": (0.72, 0.82)},   # purpura-magenta (sombra)
    "solar":  {"hue": (0.12, 0.17)},   # dorado (luz)
}
ARQUETIPOS = ["blob", "humanoide", "bestia", "fulgor"]


def _hsv_to_rgb(h: float, s: float, v: float) -> Tuple[int, int, int]:
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i % 6
    r, g, b = [
        (v, t, p), (q, v, p), (p, v, t),
        (p, q, v), (t, p, v), (v, p, q),
    ][i]
    return int(r * 255), int(g * 255), int(b * 255)


def _paleta(rng: np.random.Generator, elemento: str):
    h0, h1 = ELEMENTOS[elemento]["hue"]
    h = rng.uniform(h0, h1)
    cuerpo = _hsv_to_rgb(h, rng.uniform(0.55, 0.9), rng.uniform(0.45, 0.75))
    brillo = _hsv_to_rgb((h + rng.uniform(-0.03, 0.03)) % 1.0, rng.uniform(0.3, 0.6),
                         rng.uniform(0.9, 1.0))
    fondo_v = rng.uniform(0.05, 0.14)
    fondo = _hsv_to_rgb(h, 0.5, fondo_v)
    return cuerpo, brillo, fondo


def _silueta_simetrica(draw: ImageDraw.ImageDraw, rng: np.random.Generator,
                       S: int, arquetipo: str, color) -> None:
    """Dibuja una silueta vertical simetrica segun el arquetipo."""
    cx = S // 2
    # perfil de ancho por fila (mitad derecha), espejado luego por el dibujo simetrico
    filas = []
    if arquetipo == "blob":
        top = int(S * rng.uniform(0.18, 0.28))
        bot = int(S * rng.uniform(0.80, 0.90))
        w_max = rng.uniform(0.22, 0.34) * S
        for y in range(top, bot):
            t = (y - top) / max(1, (bot - top))
            w = w_max * np.sin(np.pi * t) ** 0.6
            filas.append((y, w))
    elif arquetipo == "humanoide":
        # cabeza + torso + piernas sugeridas
        head_r = rng.uniform(0.10, 0.15) * S
        hy = int(S * rng.uniform(0.20, 0.26))
        draw.ellipse([cx - head_r, hy - head_r, cx + head_r, hy + head_r], fill=color)
        top = int(hy + head_r * 0.6)
        bot = int(S * rng.uniform(0.82, 0.90))
        w_sh = rng.uniform(0.16, 0.24) * S
        for y in range(top, bot):
            t = (y - top) / max(1, (bot - top))
            w = w_sh * (1.0 - 0.45 * t)
            if t > 0.75:  # separacion de piernas
                w *= 0.55
            filas.append((y, w))
    elif arquetipo == "bestia":
        top = int(S * rng.uniform(0.28, 0.36))
        bot = int(S * rng.uniform(0.78, 0.88))
        w_max = rng.uniform(0.26, 0.38) * S
        for y in range(top, bot):
            t = (y - top) / max(1, (bot - top))
            w = w_max * (0.5 + 0.5 * np.cos(np.pi * (t - 0.3)))
            filas.append((y, max(2.0, w)))
    else:  # fulgor: forma de llama/gota
        top = int(S * rng.uniform(0.16, 0.24))
        bot = int(S * rng.uniform(0.82, 0.92))
        w_max = rng.uniform(0.18, 0.28) * S
        for y in range(top, bot):
            t = (y - top) / max(1, (bot - top))
            w = w_max * (t ** 0.7) * (1.0 - 0.3 * t)
            filas.append((y, max(1.5, w)))
    for y, w in filas:
        draw.rectangle([cx - w, y, cx + w, y + 1], fill=color)


def _dibujar_lumen(rng: np.random.Generator, S: int) -> Image.Image:
    elemento = rng.choice(list(ELEMENTOS.keys()))
    arquetipo = rng.choice(ARQUETIPOS)
    cuerpo, brillo, fondo = _paleta(rng, elemento)

    # fondo con vigneta tenue del color elemental
    img = Image.new("RGB", (S, S), fondo)
    draw = ImageDraw.Draw(img)

    # silueta
    _silueta_simetrica(draw, rng, S, arquetipo, cuerpo)

    # halo/glow: capa desenfocada del cuerpo
    glow = Image.new("RGB", (S, S), (0, 0, 0))
    gdraw = ImageDraw.Draw(glow)
    _silueta_simetrica(gdraw, rng, S, arquetipo, brillo)
    glow = glow.filter(ImageFilter.GaussianBlur(radius=rng.uniform(2.0, 4.0)))
    img = Image.blend(img, Image.composite(glow, img, glow.convert("L")), 0.35)
    draw = ImageDraw.Draw(img)

    cx = S // 2
    # nucleo brillante central
    cy = int(S * rng.uniform(0.45, 0.58))
    cr = rng.uniform(0.04, 0.08) * S
    draw.ellipse([cx - cr, cy - cr, cx + cr, cy + cr], fill=brillo)

    # ojos simetricos
    n_ojos = int(rng.choice([1, 2, 2, 2, 3]))
    oy = int(S * rng.uniform(0.30, 0.42))
    orr = rng.uniform(0.02, 0.045) * S
    if n_ojos == 1:
        draw.ellipse([cx - orr, oy - orr, cx + orr, oy + orr], fill=brillo)
    else:
        sep = rng.uniform(0.07, 0.13) * S
        draw.ellipse([cx - sep - orr, oy - orr, cx - sep + orr, oy + orr], fill=brillo)
        draw.ellipse([cx + sep - orr, oy - orr, cx + sep + orr, oy + orr], fill=brillo)
        if n_ojos == 3:
            draw.ellipse([cx - orr * 0.7, oy - sep * 0.6 - orr * 0.7,
                          cx + orr * 0.7, oy - sep * 0.6 + orr * 0.7], fill=brillo)

    # apendices superiores simetricos (cuernos / antenas) con probabilidad
    if rng.random() < 0.6:
        hy = int(S * rng.uniform(0.18, 0.26))
        hx = rng.uniform(0.06, 0.12) * S
        hl = rng.uniform(0.06, 0.12) * S
        draw.line([cx - hx, hy, cx - hx - hl * 0.4, hy - hl], fill=cuerpo, width=2)
        draw.line([cx + hx, hy, cx + hx + hl * 0.4, hy - hl], fill=cuerpo, width=2)

    # grano sutil para textura
    arr = np.asarray(img).astype(np.int16)
    ruido = rng.integers(-8, 9, size=arr.shape)
    arr = np.clip(arr + ruido, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def generar_dataset(out_dir: str, n: int, size: int, seed: int,
                    sample_dir: str | None = None) -> int:
    """Genera `n` imagenes del universo Lumen y las guarda en `out_dir`."""
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i in range(n):
        img = _dibujar_lumen(rng, size)
        img.save(os.path.join(out_dir, f"lumen_{i:05d}.png"))
    # guarda una rejilla de muestra del dataset para el informe
    if sample_dir is not None:
        os.makedirs(sample_dir, exist_ok=True)
        muestra = Image.new("RGB", (size * 8, size * 8))
        rng2 = np.random.default_rng(seed + 1)
        for j in range(64):
            muestra.paste(_dibujar_lumen(rng2, size), ((j % 8) * size, (j // 8) * size))
        muestra.save(os.path.join(sample_dir, "dataset_muestra_8x8.png"))
    return n


# --------------------------------------------------------------------------- #
# Dataset de PyTorch
# --------------------------------------------------------------------------- #
class LumenDataset(Dataset):
    """Carga las imagenes PNG y las normaliza a [-1, 1] (tanh del generador)."""

    def __init__(self, img_dir: str, size: int = 64):
        self.paths = sorted(
            os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".png")
        )
        self.size = size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.paths[idx]).convert("RGB").resize((self.size, self.size))
        arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0   # [-1, 1]
        return torch.from_numpy(arr).permute(2, 0, 1)            # C,H,W


def make_dataloader(img_dir: str, size: int, batch_size: int, seed: int) -> DataLoader:
    ds = LumenDataset(img_dir, size)
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True,
                      num_workers=0, generator=g)


if __name__ == "__main__":
    # Genera el dataset usando la configuracion del proyecto.
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config as C

    n = generar_dataset(C.IMG_DIR, C.Config().dataset_size, C.Config().image_size,
                        C.SEED_DATA, C.SAMPLE_DIR)
    print(f"Dataset Lumen generado: {n} imagenes en {C.IMG_DIR}")
