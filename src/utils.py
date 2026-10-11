"""Utilidades compartidas: conversion de tensores, rejillas de imagenes y curvas."""
from __future__ import annotations

import os
from typing import List

import numpy as np
import torch
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def tensor_to_uint8(x: torch.Tensor) -> np.ndarray:
    """(C,H,W) en [-1,1] -> (H,W,C) uint8 en [0,255]."""
    x = x.detach().cpu().clamp(-1, 1)
    x = (x + 1.0) / 2.0
    arr = (x.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return arr


def grid_image(batch: torch.Tensor, nrow: int = 4, pad: int = 2,
               bg: int = 20) -> Image.Image:
    """Arma una rejilla PIL a partir de un lote (B,C,H,W)."""
    b, c, h, w = batch.shape
    ncol = nrow
    nrows = (b + ncol - 1) // ncol
    W = ncol * w + (ncol + 1) * pad
    H = nrows * h + (nrows + 1) * pad
    canvas = Image.new("RGB", (W, H), (bg, bg, bg))
    for i in range(b):
        r, cc = divmod(i, ncol)
        tile = Image.fromarray(tensor_to_uint8(batch[i]))
        x = pad + cc * (w + pad)
        y = pad + r * (h + pad)
        canvas.paste(tile, (x, y))
    return canvas


def save_grid(batch: torch.Tensor, path: str, nrow: int = 4) -> None:
    grid_image(batch, nrow=nrow).save(path)


def montage_evolucion(grids: List[Image.Image], etiquetas: List[str],
                      path: str) -> None:
    """Pega varias rejillas de ruido fijo (una por epoca) en una sola figura."""
    if not grids:
        return
    gw, gh = grids[0].size
    label_h = 22
    cols = len(grids)
    canvas = Image.new("RGB", (cols * gw + (cols + 1) * 6,
                               gh + label_h + 12), (12, 12, 16))
    draw = ImageDraw.Draw(canvas)
    for i, (g, et) in enumerate(zip(grids, etiquetas)):
        x = 6 + i * (gw + 6)
        draw.text((x + 4, 4), et, fill=(230, 230, 230))
        canvas.paste(g, (x, label_h))
    canvas.save(path)


def plot_curvas(history: dict, path: str, titulo: str) -> None:
    """Dibuja curvas de perdida de G y D y la salida media de D."""
    ep = history["epoch"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))

    ax[0].plot(ep, history["loss_d"], label="Perdida D", color="#d1495b")
    ax[0].plot(ep, history["loss_g"], label="Perdida G", color="#2e8bc0")
    ax[0].set_xlabel("Epoca"); ax[0].set_ylabel("Perdida")
    ax[0].set_title("Perdidas G vs D"); ax[0].legend(); ax[0].grid(alpha=0.3)

    ax[1].plot(ep, history["d_real"], label="D(x) reales", color="#2a9d8f")
    ax[1].plot(ep, history["d_fake"], label="D(G(z)) falsas", color="#e9c46a")
    ax[1].axhline(0.5, ls="--", c="gray", alpha=0.6)
    ax[1].set_xlabel("Epoca"); ax[1].set_ylabel("Prob. media (sigmoid)")
    ax[1].set_title("Salida media del discriminador"); ax[1].legend(); ax[1].grid(alpha=0.3)

    fig.suptitle(titulo)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_comparacion(histories: dict, metric: str, path: str, titulo: str,
                     ylabel: str) -> None:
    """Compara una metrica entre varios experimentos (p.ej. base vs variante)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for nombre, h in histories.items():
        ax.plot(h["epoch"], h[metric], label=nombre)
    ax.set_xlabel("Epoca"); ax.set_ylabel(ylabel)
    ax.set_title(titulo); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
