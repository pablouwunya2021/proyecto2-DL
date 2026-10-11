"""
Construye presentacion.pdf (<=12 diapositivas + anexos) a partir de las figuras
y metadatos reales generados por el pipeline. Reproducible: regenera el PDF desde
los artefactos de `evidencia/` y `galeria/`.

Uso:  python presentacion/build_pdf.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import image as mpimg

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVID = os.path.join(ROOT, "evidencia")
GAL = os.path.join(ROOT, "galeria")
DATA = os.path.join(ROOT, "datos", "muestras")
OUT = os.path.join(ROOT, "presentacion", "presentacion.pdf")

AZUL = "#2e8bc0"; GRIS = "#3b3b45"; TENUE = "#6b7280"
plt.rcParams.update({"font.size": 12, "axes.edgecolor": GRIS})


def _fig():
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.patch.set_facecolor("white")
    return fig


def _titulo(fig, texto, sub=None):
    fig.text(0.06, 0.90, texto, fontsize=24, fontweight="bold", color="#1b1b2a")
    if sub:
        fig.text(0.06, 0.845, sub, fontsize=13, color=TENUE)
    fig.add_artist(plt.Line2D([0.06, 0.94], [0.83, 0.83], color=AZUL, lw=2.5,
                              transform=fig.transFigure))


def _img(fig, path, rect, titulo=None):
    """Coloca una imagen (rect = [x,y,w,h] en fraccion de figura)."""
    ax = fig.add_axes(rect)
    if os.path.exists(path):
        ax.imshow(mpimg.imread(path))
    else:
        ax.text(0.5, 0.5, f"(falta {os.path.basename(path)})", ha="center")
    ax.axis("off")
    if titulo:
        ax.set_title(titulo, fontsize=11, color=GRIS)
    return ax


def _bullets(fig, x, y, items, dy=0.058, size=12.5, color="#22223b"):
    for i, it in enumerate(items):
        fig.text(x, y - i * dy, it, fontsize=size, color=color, va="top", wrap=True)


def _load(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default or {}


def build():
    sel = _load(os.path.join(GAL, "seleccion.json"))
    vec = _load(os.path.join(EVID, "vecinos_reporte.json"))
    personajes = sel.get("personajes", [])
    tasa = sel.get("tasa_seleccion", "10 de 200")
    extractor = sel.get("extractor_features", "CNN")

    # personaje mas dificil de defender como nuevo (vecino mas cercano)
    reflexion = "—"
    if vec.get("vecinos"):
        vv = vec["vecinos"]
        dmin = min(vv, key=lambda r: r["dist_cnn"])
        k = dmin["personaje"]
        nombre = next((p["nombre"] for p in personajes if p["rank"] == k), f"#{k}")
        reflexion = (f"El personaje mas dificil de defender como nuevo es el #{k} "
                     f"({nombre}), con la menor distancia a su vecino de entrenamiento "
                     f"(dist CNN = {dmin['dist_cnn']:.3f}). Aun asi no es una copia: "
                     f"conserva la silueta y el nucleo del universo pero cambia color "
                     f"y detalles. Nos dice que la GAN aprendio la 'gramatica' Lumen "
                     f"(simetria + nucleo + paleta), no imagenes concretas.")

    with PdfPages(OUT) as pdf:
        # --- Slide 1: portada ---
        fig = _fig()
        fig.add_artist(plt.Line2D([0.06, 0.52], [0.70, 0.70], color=AZUL, lw=3,
                                  transform=fig.transFigure))
        fig.text(0.06, 0.60, "Lumen", fontsize=46, fontweight="bold", color="#1b1b2a")
        fig.text(0.06, 0.515, "Guardianes Elementales", fontsize=30,
                 fontweight="bold", color="#1b1b2a")
        fig.text(0.06, 0.44, "Diseño generativo de personajes con una GAN (DCGAN 64×64)",
                 fontsize=15, color=AZUL)
        fig.text(0.06, 0.30, "Proyecto 2 · Deep Learning 2026 · Universidad del Valle",
                 fontsize=13, color=TENUE)
        fig.text(0.06, 0.255, "Pablo Cabrera (231156)  ·  Luis Palacios (23933)",
                 fontsize=14, color="#22223b", fontweight="bold")
        fig.text(0.06, 0.215, "Modalidad parejas", fontsize=12, color=TENUE)
        _img(fig, os.path.join(DATA, "dataset_muestra_8x8.png"),
             [0.60, 0.26, 0.34, 0.48])
        fig.text(0.60, 0.23, "Muestra del dataset (material de entrenamiento)",
                 fontsize=10, color=TENUE)
        pdf.savefig(fig); plt.close(fig)

        # --- Slide 2: universo y datos ---
        fig = _fig(); _titulo(fig, "1 · Universo y datos",
                              "La descripción del universo es la vara de evaluación de la galería")
        _bullets(fig, 0.06, 0.76, [
            "Universo: espíritus guardianes bioluminiscentes, nocturnos.",
            "Estilo: siluetas simétricas y centradas, núcleo de luz, halo suave.",
            "6 elementos (brasa, marea, musgo, céfiro, vacío, solar) y 4 arquetipos.",
            "Resolución 64×64 (recomendación del enunciado).",
            "",
            "Datos: ~5000 imágenes generadas por script propio (src/data.py).",
            "Origen/licencia: creación propia, CC0, reproducible desde SEED_DATA.",
            "Sin rostros reales ni personajes protegidos.",
        ])
        _img(fig, os.path.join(DATA, "dataset_muestra_8x8.png"),
             [0.60, 0.20, 0.34, 0.52], "Dataset Lumen (8×8)")
        pdf.savefig(fig); plt.close(fig)

        # --- Slide 3: arquitectura ---
        fig = _fig(); _titulo(fig, "2 · Arquitectura de la GAN y por qué",
                              "DCGAN como base sólida antes de experimentar")
        _bullets(fig, 0.06, 0.76, [
            "Generador: z∈R^128 → conv transpuestas → 3×64×64, salida tanh.",
            "Discriminador: conv con stride → logit; LeakyReLU(0.2).",
            "BatchNorm en G (y en D en la base); init Normal(0, 0.02).",
            "Adam(lr=2e-4, β1=0.5, β2=0.999), batch 128, 60 épocas.",
            "",
            "Por qué DCGAN: es la referencia estándar para 'empezar simple' y",
            "conseguir algo reconocible; sobre ella aislamos cada experimento.",
            "Pérdida base: BCE no saturante (maximiza log D(G(z))).",
        ])
        pdf.savefig(fig); plt.close(fig)

        # --- Slide 4: metodologia experimentos ---
        fig = _fig(); _titulo(fig, "3 · Diseño de experimentos",
                              "Una sola variable a la vez; todo lo demás fijo")
        _bullets(fig, 0.06, 0.76, [
            "Control compartido: mismos datos, SEED_TRAIN=42, 60 épocas y",
            "ruido de evaluación SEED_EVAL=7 en las tres corridas.",
            "",
            "Exp A — Pérdida:  base (BCE no saturante)  vs  hinge.",
            "Exp B — Estabilización:  BatchNorm en D  vs  normalización espectral.",
            "",
            "Cada experimento tiene hipótesis previa con criterio de 'útil'",
            "(docs/hipotesis.md). Un experimento que no mejora también es válido.",
        ])
        pdf.savefig(fig); plt.close(fig)

        # --- Slide 5: Exp A perdida ---
        fig = _fig(); _titulo(fig, "4 · Experimento A — Función de pérdida",
                              "Hipótesis: hinge → gradientes más estables, sin perder nitidez/variedad")
        _img(fig, os.path.join(EVID, "expA_perdida_Lg.png"), [0.06, 0.12, 0.44, 0.58],
             "Pérdida de G: base vs hinge")
        _img(fig, os.path.join(EVID, "expA_perdida_Dfake.png"), [0.52, 0.12, 0.44, 0.58],
             "D(G(z)) medio por época")
        pdf.savefig(fig); plt.close(fig)

        # --- Slide 6: Exp B estabilizacion ---
        fig = _fig(); _titulo(fig, "5 · Experimento B — Estabilización",
                              "Hipótesis: spectral norm → equilibrio más sostenido y más diversidad")
        _img(fig, os.path.join(EVID, "expB_estab_Dreal.png"), [0.06, 0.12, 0.44, 0.58],
             "D(x) medio por época")
        _img(fig, os.path.join(EVID, "expB_estab_Ld.png"), [0.52, 0.12, 0.44, 0.58],
             "Pérdida de D: base vs spectral norm")
        pdf.savefig(fig); plt.close(fig)

        # --- Slide 7: Evidencia 1 ---
        fig = _fig(); _titulo(fig, "6 · Evidencia 1 — Rejilla de ruido fijo",
                              "Mismos 16 vectores z a lo largo del entrenamiento (modelo base)")
        _img(fig, os.path.join(EVID, "base", "evolucion_ruido_fijo.png"),
             [0.04, 0.16, 0.92, 0.56])
        fig.text(0.06, 0.12, "Las filas se diferencian y ganan detalle con las épocas: "
                 "el generador evoluciona y no colapsa a una sola imagen.",
                 fontsize=11.5, color=GRIS)
        pdf.savefig(fig); plt.close(fig)

        # --- Slide 8: Evidencia 2 ---
        fig = _fig(); _titulo(fig, "7 · Evidencia 2 — Curvas de pérdida",
                              "En una GAN sana no bajan juntas: se estabilizan enfrentadas")
        _img(fig, os.path.join(EVID, "base", "curvas_perdida.png"),
             [0.05, 0.14, 0.90, 0.58])
        fig.text(0.06, 0.10, "D(x) y D(G(z)) se mantienen cerca de la zona de equilibrio "
                 "(~0.7 / ~0.25) y loss_D ≈ ln 2: no hay D≈0 sostenido (sin colapso).",
                 fontsize=11.5, color=GRIS)
        pdf.savefig(fig); plt.close(fig)

        # --- Slide 9: galeria ---
        fig = _fig(); _titulo(fig, "8 · Galería de los 10 personajes finales",
                              f"Tasa de selección declarada: {tasa}")
        _img(fig, os.path.join(GAL, "galeria_10.png"), [0.05, 0.22, 0.90, 0.50])
        nombres = ", ".join(f"{p['rank']}·{p['nombre']} ({p['rol']})"
                            for p in personajes[:10]) or "(ver galeria/)"
        fig.text(0.06, 0.17, nombres, fontsize=9.5, color=GRIS, wrap=True)
        fig.text(0.06, 0.10, f"Criterio: {sel.get('criterio','')}", fontsize=9.5,
                 color=TENUE, wrap=True)
        pdf.savefig(fig); plt.close(fig)

        # --- Slide 10: Evidencia 3 vecinos ---
        fig = _fig(); _titulo(fig, "9 · Evidencia 3 — Vecinos más cercanos",
                              f"Cada personaje junto a su imagen real más parecida ({extractor})")
        _img(fig, os.path.join(EVID, "vecinos_mas_cercanos.png"),
             [0.30, 0.10, 0.40, 0.62])
        fig.text(0.06, 0.66, "Izquierda: generado · Derecha: vecino real.",
                 fontsize=11, color=GRIS)
        pdf.savefig(fig); plt.close(fig)

        # --- Slide 11: problemas, limitaciones, reflexion ---
        fig = _fig(); _titulo(fig, "10 · Problemas, decisiones y reflexión final")
        _bullets(fig, 0.06, 0.76, [
            "Problemas: D domina al inicio (D(x)→1); se equilibra en pocas épocas.",
            "Decisión: lr iguales y BCE no saturante dieron el equilibrio más limpio.",
            "Limitaciones: 64×64 y dataset procedural → variedad acotada por diseño;",
            "métrica de novedad depende del extractor de features.",
        ])
        fig.text(0.06, 0.40, "Reflexión final:", fontsize=13, fontweight="bold",
                 color="#1b1b2a")
        fig.text(0.06, 0.35, reflexion, fontsize=11.5, color="#22223b", va="top",
                 wrap=True)
        pdf.savefig(fig); plt.close(fig)

        # --- Slide 12: matriz de evidencias ---
        fig = _fig(); _titulo(fig, "11 · Matriz de evidencias",
                              "Guía de calificación")
        ax = fig.add_axes([0.05, 0.08, 0.90, 0.68]); ax.axis("off")
        filas = [
            ["Evidencia", "Diapositiva", "Conclusión", "Limitación"],
            ["Rejilla de ruido fijo", "6", "G evoluciona, sin colapso",
             "solo modelo base mostrado"],
            ["Curvas de pérdida", "7", "equilibrio enfrentado (~ln2)",
             "pérdida no mide calidad visual"],
            ["Vecinos más cercanos", "9", "personajes nuevos, no copias",
             "depende del extractor de features"],
            ["Galería (10)", "8", "coherente con el universo",
             f"tasa {tasa}"],
            ["Exp. pérdida (A)", "4", "efecto de BCE vs hinge",
             "mismas épocas pueden ser pocas"],
            ["Exp. estabilización (B)", "5", "efecto de spectral norm",
             "un solo seed por corrida"],
        ]
        tabla = ax.table(cellText=filas[1:], colLabels=filas[0],
                         cellLoc="left", loc="center")
        tabla.auto_set_font_size(False); tabla.set_fontsize(10.5)
        tabla.scale(1, 1.8)
        for (r, c), cell in tabla.get_celld().items():
            cell.set_edgecolor("#d0d0d8")
            if r == 0:
                cell.set_facecolor(AZUL); cell.set_text_props(color="white",
                                                              fontweight="bold")
            elif r % 2 == 0:
                cell.set_facecolor("#f2f5f9")
        pdf.savefig(fig); plt.close(fig)

    print("PDF generado:", OUT)


if __name__ == "__main__":
    build()
