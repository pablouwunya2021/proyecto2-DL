"""
Evidencia 3: prueba de vecinos mas cercanos.

Para cada personaje de la galeria busca la imagen de entrenamiento mas parecida
y las muestra lado a lado (estilo figura 2 de Goodfellow et al., 2014). Reporta
la distancia en el espacio de la CNN y tambien en pixeles, para distinguir un
generador que aprendio de uno que memorizo.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
from PIL import Image, ImageDraw

import config as C
from src.models import Generator
from src.features import FeatureExtractor
from src.data import LumenDataset
from src.utils import tensor_to_uint8
from src.gallery import GALLERY_SEED, N_MUESTRAS


def _load_z():
    d = np.load(os.path.join(C.GALLERY_DIR, "z_vectors.npz"))
    return d["z"], str(d["modelo"]), d["indices"]


def prueba_vecinos(modelo: str = "base"):
    device = C.get_device()
    cfg = C.base_config()
    G = Generator(cfg.nz, cfg.ngf, cfg.channels).to(device)
    G.load_state_dict(torch.load(os.path.join(C.CKPT_DIR, f"G_{modelo}.pt"),
                                 map_location=device))
    G.eval()
    fx = FeatureExtractor(device)

    z, _, _ = _load_z()
    z = torch.from_numpy(z).float().view(z.shape[0], cfg.nz, 1, 1).to(device)
    with torch.no_grad():
        gen = G(z)
    gen_feats = fx(gen).cpu().numpy()
    gen_px = ((gen.clamp(-1, 1) + 1) / 2).cpu().numpy().reshape(len(z), -1)

    # todo el dataset como banco de vecinos
    ds = LumenDataset(C.IMG_DIR, cfg.image_size)
    all_imgs = torch.stack([ds[i] for i in range(len(ds))])
    tf = []
    for i in range(0, len(all_imgs), 256):
        tf.append(fx(all_imgs[i:i + 256]).cpu().numpy())
    train_feats = np.concatenate(tf, 0)
    train_px = ((all_imgs + 1) / 2).numpy().reshape(len(all_imgs), -1)

    filas, reporte = [], []
    for k in range(len(z)):
        # vecino por caracteristicas de la CNN (criterio principal)
        d_feat = np.linalg.norm(train_feats - gen_feats[k], axis=1)
        j_feat = int(np.argmin(d_feat))
        # vecino por pixeles (referencia)
        d_px = np.linalg.norm(train_px - gen_px[k], axis=1)
        j_px = int(np.argmin(d_px))

        gen_img = Image.fromarray(tensor_to_uint8(gen[k])).resize((96, 96), Image.NEAREST)
        nn_img = Image.fromarray(tensor_to_uint8(ds[j_feat])).resize((96, 96), Image.NEAREST)
        fila = Image.new("RGB", (96 * 2 + 12, 96 + 20), (16, 16, 20))
        fila.paste(gen_img, (0, 18)); fila.paste(nn_img, (96 + 12, 18))
        dd = ImageDraw.Draw(fila)
        dd.text((2, 4), "generado", fill=(230, 230, 230))
        dd.text((96 + 14, 4), "vecino real", fill=(180, 200, 255))
        filas.append(fila)
        reporte.append({
            "personaje": k + 1,
            "vecino_cnn_idx": j_feat, "dist_cnn": float(d_feat[j_feat]),
            "vecino_pixel_idx": j_px, "dist_pixel": float(d_px[j_px]),
        })

    # figura unica lado a lado (2 columnas)
    fw, fh = filas[0].size
    cols = 2
    rows = (len(filas) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * fw + (cols + 1) * 8,
                               rows * fh + (rows + 1) * 8), (10, 10, 14))
    for i, fila in enumerate(filas):
        r, cc = divmod(i, cols)
        canvas.paste(fila, (8 + cc * (fw + 8), 8 + r * (fh + 8)))
    out = os.path.join(C.EVID_DIR, "vecinos_mas_cercanos.png")
    canvas.save(out)

    with open(os.path.join(C.EVID_DIR, "vecinos_reporte.json"), "w") as f:
        json.dump({"extractor": fx.mode, "vecinos": reporte}, f, indent=2)

    dcnn = np.array([r["dist_cnn"] for r in reporte])
    print(f"Vecinos mas cercanos -> {out} | features={fx.mode}")
    print(f"dist CNN: min={dcnn.min():.3f} media={dcnn.mean():.3f} max={dcnn.max():.3f}")
    print("El personaje con vecino MAS cercano (mas dificil de defender como nuevo) "
          f"es el #{int(np.argmin(dcnn))+1} (dist={dcnn.min():.3f}).")
    return reporte


if __name__ == "__main__":
    import sys
    prueba_vecinos(sys.argv[1] if len(sys.argv) > 1 else "base")
