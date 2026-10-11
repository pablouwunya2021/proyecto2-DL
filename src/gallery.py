"""
Genera la galeria final de personajes a partir del generador entrenado.

Estrategia de seleccion (declarada, requisito de "tasa de seleccion"):
 1. Se muestrean N=200 vectores z con una semilla fija (GALLERY_SEED).
 2. Se descartan los que quedan demasiado cerca de su vecino de entrenamiento
    (poco novedosos) y se penalizan los casi duplicados entre si.
 3. Entre el resto se eligen 10 personajes lo mas diversos posible con un
    criterio de punto-mas-lejano (farthest-point) en el espacio de caracteristicas.

Cada personaje se guarda como PNG junto con su vector z (galeria/z_vectors.npz),
de modo que el codigo puede regenerar exactamente la galeria.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
from PIL import Image

import config as C
from src.models import Generator
from src.features import FeatureExtractor
from src.data import LumenDataset
from src.utils import tensor_to_uint8, grid_image

GALLERY_SEED = 2026
N_MUESTRAS = 200
N_FINAL = 10

# Nombres y roles por elemento (coherencia de universo; aporta a la bonificacion).
NOMBRES = {
    "brasa":  ("Ignar", "Pyra", "Emberil", "Cindros"),
    "marea":  ("Nereo", "Undina", "Marel", "Tidon"),
    "musgo":  ("Verdal", "Mycel", "Sylfa", "Bryos"),
    "cefiro": ("Zephyr", "Aeli", "Ventis", "Nimba"),
    "vacio":  ("Umbryx", "Noctis", "Vael", "Shadra"),
    "solar":  ("Lumis", "Solenne", "Aurex", "Helio"),
}
ROLES = {
    "brasa": "Centinela de forja", "marea": "Guardian de mareas",
    "musgo": "Druida de raiz", "cefiro": "Heraldo del viento",
    "vacio": "Acechante umbrio", "solar": "Portador de luz",
}
# Centros de tono (HSV) por elemento, para clasificar un personaje generado.
HUE_CENTRO = {"brasa": 0.04, "marea": 0.57, "musgo": 0.33,
              "cefiro": 0.48, "vacio": 0.77, "solar": 0.14}


def _elemento_de(img_uint8: np.ndarray) -> str:
    """Clasifica el elemento por el tono dominante de los pixeles brillantes."""
    import colorsys
    arr = img_uint8.astype(np.float32) / 255.0
    bril = arr.reshape(-1, 3)
    lum = bril.mean(1)
    sel = bril[lum > np.percentile(lum, 75)]  # pixeles mas luminosos (la criatura)
    if len(sel) == 0:
        sel = bril
    hs = np.array([colorsys.rgb_to_hsv(*p)[0] for p in sel[::7]])
    hmean = np.median(hs)
    best = min(HUE_CENTRO, key=lambda k: min(abs(hmean - HUE_CENTRO[k]),
                                             1 - abs(hmean - HUE_CENTRO[k])))
    return best


def _load_generator(name: str, device) -> Generator:
    cfg = C.base_config()
    G = Generator(cfg.nz, cfg.ngf, cfg.channels).to(device)
    G.load_state_dict(torch.load(os.path.join(C.CKPT_DIR, f"G_{name}.pt"),
                                 map_location=device))
    G.eval()
    return G


def _farthest_point(feats: np.ndarray, k: int, orden0: int) -> list:
    """Selecciona k indices maximizando la distancia minima entre elegidos."""
    elegidos = [orden0]
    d = np.linalg.norm(feats - feats[orden0], axis=1)
    while len(elegidos) < k:
        nxt = int(np.argmax(d))
        elegidos.append(nxt)
        d = np.minimum(d, np.linalg.norm(feats - feats[nxt], axis=1))
    return elegidos


def generar_galeria(modelo: str = "base"):
    device = C.get_device()
    G = _load_generator(modelo, device)
    fx = FeatureExtractor(device)

    # 1) muestreo reproducible
    g = torch.Generator().manual_seed(GALLERY_SEED)
    z = torch.randn(N_MUESTRAS, C.base_config().nz, 1, 1, generator=g).to(device)
    with torch.no_grad():
        fake = G(z)
    feats = fx(fake).cpu().numpy()

    # 2) novedad: distancia al vecino de entrenamiento mas cercano (en features)
    ds = LumenDataset(C.IMG_DIR, C.base_config().image_size)
    idx_train = np.random.default_rng(0).choice(len(ds), size=1500, replace=False)
    train_batch = torch.stack([ds[i] for i in idx_train])
    tf = []
    for i in range(0, len(train_batch), 256):
        tf.append(fx(train_batch[i:i + 256]).cpu().numpy())
    train_feats = np.concatenate(tf, 0)
    nn_dist = np.array([np.min(np.linalg.norm(train_feats - f, axis=1)) for f in feats])

    # 3) candidatos con suficiente novedad y seleccion por punto-mas-lejano
    umbral = np.percentile(nn_dist, 40)        # descarta el 40% menos novedoso
    cand = np.where(nn_dist >= umbral)[0]
    if len(cand) < N_FINAL:
        cand = np.argsort(-nn_dist)[:max(N_FINAL * 3, 30)]
    orden0 = int(cand[np.argmax(nn_dist[cand])])
    sel_local = _farthest_point(feats[cand], N_FINAL, list(cand).index(orden0))
    seleccion = [int(cand[i]) for i in sel_local]

    # 4) guardar galeria + vectores z + metadatos
    os.makedirs(C.GALLERY_DIR, exist_ok=True)
    z_np = z.detach().cpu().numpy().reshape(N_MUESTRAS, -1)
    meta = []
    usados = {}
    for rank, j in enumerate(seleccion, 1):
        img = tensor_to_uint8(fake[j])
        elem = _elemento_de(img)
        pool = NOMBRES[elem]
        usados[elem] = usados.get(elem, 0)
        nombre = pool[usados[elem] % len(pool)]
        usados[elem] += 1
        fname = f"personaje_{rank:02d}_{nombre}.png"
        Image.fromarray(img).resize((256, 256), Image.NEAREST).save(
            os.path.join(C.GALLERY_DIR, fname))
        meta.append({
            "rank": rank, "nombre": nombre, "elemento": elem,
            "rol": ROLES[elem], "indice_muestra": j,
            "gallery_seed": GALLERY_SEED, "archivo": fname,
            "dist_vecino_entrenamiento": float(nn_dist[j]),
        })

    np.savez(os.path.join(C.GALLERY_DIR, "z_vectors.npz"),
             z=z_np[seleccion], indices=np.array(seleccion),
             gallery_seed=GALLERY_SEED, modelo=modelo)

    seleccion_info = {
        "modelo": modelo,
        "muestras_generadas": N_MUESTRAS,
        "personajes_elegidos": N_FINAL,
        "tasa_seleccion": f"{N_FINAL} de {N_MUESTRAS}",
        "criterio": ("Se descarto el 40% de muestras mas cercanas a su vecino de "
                     "entrenamiento (menos novedosas) y entre el resto se eligieron "
                     "10 maximizando la diversidad (punto-mas-lejano en el espacio "
                     "de caracteristicas de la CNN)."),
        "extractor_features": fx.mode,
        "personajes": meta,
    }
    with open(os.path.join(C.GALLERY_DIR, "seleccion.json"), "w") as f:
        json.dump(seleccion_info, f, indent=2, ensure_ascii=False)

    # hoja de contacto de los 10
    grid_image(fake[seleccion], nrow=5).resize((5 * 128, 2 * 128), Image.NEAREST).save(
        os.path.join(C.GALLERY_DIR, "galeria_10.png"))

    print(f"Galeria generada ({modelo}): 10 personajes en {C.GALLERY_DIR} "
          f"| features={fx.mode} | tasa=10/{N_MUESTRAS}")
    return seleccion_info


if __name__ == "__main__":
    import sys
    generar_galeria(sys.argv[1] if len(sys.argv) > 1 else "base")
