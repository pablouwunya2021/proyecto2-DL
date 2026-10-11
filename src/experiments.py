"""
Orquesta los dos experimentos obligatorios respetando la regla de cambiar UNA
sola cosa a la vez respecto a la configuracion base (mismos datos, semilla,
epocas y ruido de evaluacion).

  Experimento A (perdida):        base (BCE no saturante)  vs  hinge
  Experimento B (estabilizacion): base (BatchNorm en D)    vs  spectral norm en D

Las hipotesis previas estan en docs/hipotesis.md y se reproducen aqui como guia.
"""
from __future__ import annotations

import json
import os

import config as C
from src.train import entrenar
from src.utils import plot_comparacion


def _cfg(name, **kw) -> C.Config:
    cfg = C.base_config()
    cfg.name = name
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


# Las tres corridas necesarias (base se reutiliza como control de A y B).
RUNS = {
    "base":     _cfg("base"),                       # BCE no saturante, BatchNorm en D
    "hinge":    _cfg("hinge", loss="hinge"),        # Exp A: solo cambia la perdida
    "spectral": _cfg("spectral", spectral_norm=True),  # Exp B: solo cambia la norm. de D
}


def _load_history(name: str) -> dict:
    with open(os.path.join(C.EVID_DIR, name, "history.json")) as f:
        return json.load(f)["history"]


def entrenar_todo():
    for name, cfg in RUNS.items():
        print(f"\n===== Entrenando: {cfg.describe()} =====", flush=True)
        entrenar(cfg)


def comparar():
    """Genera las figuras de comparacion de ambos experimentos."""
    H = {name: _load_history(name) for name in RUNS}

    # Experimento A: perdida (base vs hinge)
    A = {"base (BCE)": H["base"], "hinge": H["hinge"]}
    plot_comparacion(A, "d_fake", os.path.join(C.EVID_DIR, "expA_perdida_Dfake.png"),
                     "Exp. A - perdida: D(G(z)) por epoca", "D(G(z)) medio")
    plot_comparacion(A, "loss_g", os.path.join(C.EVID_DIR, "expA_perdida_Lg.png"),
                     "Exp. A - perdida: perdida de G", "Perdida G")

    # Experimento B: estabilizacion (base vs spectral norm)
    B = {"base (BatchNorm)": H["base"], "spectral norm": H["spectral"]}
    plot_comparacion(B, "d_real", os.path.join(C.EVID_DIR, "expB_estab_Dreal.png"),
                     "Exp. B - estabilizacion: D(x) por epoca", "D(x) medio")
    plot_comparacion(B, "loss_d", os.path.join(C.EVID_DIR, "expB_estab_Ld.png"),
                     "Exp. B - estabilizacion: perdida de D", "Perdida D")
    print("Figuras de comparacion guardadas en", C.EVID_DIR)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "comparar":
        comparar()
    else:
        entrenar_todo()
        comparar()
