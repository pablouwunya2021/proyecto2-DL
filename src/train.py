"""
Loop de entrenamiento de la GAN con registro de las evidencias obligatorias:

  (1) Rejilla de ruido fijo: se guarda el mismo lote z (SEED_EVAL) en varias
      epocas y se arma un montaje de evolucion.
  (2) Curvas de perdida: se registran loss_D, loss_G y las salidas medias
      D(x) y D(G(z)) por epoca.
  Ademas guarda checkpoints del generador para poder regenerar la galeria.

Uso:
    python -m src.train --name base --loss bce
    python -m src.train --name hinge --loss hinge
    python -m src.train --name spectral --spectral-norm
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.optim as optim

import config as C
from src import losses as L
from src.data import make_dataloader
from src.models import build_models
from src.utils import save_grid, grid_image, montage_evolucion, plot_curvas


def entrenar(cfg: C.Config, verbose: bool = True) -> dict:
    device = C.get_device()
    C.set_seed(C.SEED_TRAIN)

    run_dir = os.path.join(C.EVID_DIR, cfg.name)
    grids_dir = os.path.join(run_dir, "rejillas")
    os.makedirs(grids_dir, exist_ok=True)

    loader = make_dataloader(C.IMG_DIR, cfg.image_size, cfg.batch_size, C.SEED_TRAIN)
    G, D = build_models(cfg, device)

    optG = optim.Adam(G.parameters(), lr=cfg.lr_g, betas=(cfg.beta1, cfg.beta2))
    optD = optim.Adam(D.parameters(), lr=cfg.lr_d, betas=(cfg.beta1, cfg.beta2))

    # Ruido de evaluacion FIJO (misma semilla en todos los experimentos).
    g_eval = torch.Generator().manual_seed(C.SEED_EVAL)
    fixed_noise = torch.randn(cfg.fixed_noise_n, cfg.nz, 1, 1, generator=g_eval).to(device)

    history = {"epoch": [], "loss_d": [], "loss_g": [], "d_real": [], "d_fake": []}
    grids_evol, etiquetas = [], []

    def snapshot(ep_label: str):
        G.eval()
        with torch.no_grad():
            fake = G(fixed_noise)
        G.train()
        g = grid_image(fake, nrow=4)
        g.save(os.path.join(grids_dir, f"epoca_{ep_label}.png"))
        grids_evol.append(g)
        etiquetas.append(f"ep {ep_label}")

    snapshot("000")  # estado inicial (antes de entrenar)

    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        acc = {"loss_d": 0.0, "loss_g": 0.0, "d_real": 0.0, "d_fake": 0.0, "n": 0}
        for real in loader:
            real = real.to(device)
            bs = real.size(0)

            # ------------------ Discriminador ------------------
            for _ in range(cfg.n_critic):
                z = torch.randn(bs, cfg.nz, 1, 1, device=device)
                fake = G(z).detach()
                d_real = D(real)
                d_fake = D(fake)
                if cfg.loss == "bce":
                    loss_d = L.d_loss_bce(d_real, d_fake, cfg.label_smoothing)
                elif cfg.loss == "hinge":
                    loss_d = L.d_loss_hinge(d_real, d_fake)
                elif cfg.loss == "wgan-gp":
                    gp = L.gradient_penalty(D, real, fake, device)
                    loss_d = L.d_loss_wgan(d_real, d_fake) + cfg.gp_lambda * gp
                else:
                    raise ValueError(cfg.loss)
                optD.zero_grad(set_to_none=True)
                loss_d.backward()
                optD.step()

            # ------------------ Generador ------------------
            z = torch.randn(bs, cfg.nz, 1, 1, device=device)
            fake = G(z)
            d_fake_g = D(fake)
            if cfg.loss == "bce":
                loss_g = L.g_loss_bce(d_fake_g)
            elif cfg.loss == "hinge":
                loss_g = L.g_loss_hinge(d_fake_g)
            else:
                loss_g = L.g_loss_wgan(d_fake_g)
            optG.zero_grad(set_to_none=True)
            loss_g.backward()
            optG.step()

            acc["loss_d"] += loss_d.item() * bs
            acc["loss_g"] += loss_g.item() * bs
            # Para interpretar: probabilidad media que D asigna (sigmoid del logit).
            acc["d_real"] += torch.sigmoid(d_real).mean().item() * bs
            acc["d_fake"] += torch.sigmoid(d_fake).mean().item() * bs
            acc["n"] += bs

        n = acc["n"]
        history["epoch"].append(epoch)
        history["loss_d"].append(acc["loss_d"] / n)
        history["loss_g"].append(acc["loss_g"] / n)
        history["d_real"].append(acc["d_real"] / n)
        history["d_fake"].append(acc["d_fake"] / n)

        if verbose:
            print(f"[{cfg.name}] ep {epoch:3d}/{cfg.epochs} "
                  f"Ld={history['loss_d'][-1]:.3f} Lg={history['loss_g'][-1]:.3f} "
                  f"D(x)={history['d_real'][-1]:.2f} D(G)={history['d_fake'][-1]:.2f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

        if epoch % cfg.sample_every == 0 or epoch == cfg.epochs:
            snapshot(f"{epoch:03d}")
            torch.save(G.state_dict(), os.path.join(C.CKPT_DIR, f"G_{cfg.name}.pt"))

    # Evidencias finales del run.
    montage_evolucion(grids_evol, etiquetas,
                      os.path.join(run_dir, "evolucion_ruido_fijo.png"))
    plot_curvas(history, os.path.join(run_dir, "curvas_perdida.png"),
                f"Curvas de perdida - {cfg.describe()}")
    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump({"config": cfg.__dict__, "history": history}, f, indent=2)
    torch.save(G.state_dict(), os.path.join(C.CKPT_DIR, f"G_{cfg.name}.pt"))

    if verbose:
        print(f"[{cfg.name}] listo en {time.time()-t0:.0f}s -> {run_dir}")
    return history


def _cfg_from_args(args) -> C.Config:
    cfg = C.base_config()
    cfg.name = args.name
    cfg.loss = args.loss
    cfg.spectral_norm = args.spectral_norm
    cfg.label_smoothing = args.label_smoothing
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.loss == "wgan-gp":
        cfg.n_critic = 5
    return cfg


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="base")
    p.add_argument("--loss", default="bce", choices=["bce", "hinge", "wgan-gp"])
    p.add_argument("--spectral-norm", action="store_true")
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=None)
    entrenar(_cfg_from_args(p.parse_args()))
