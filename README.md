# Proyecto 2 — Diseño generativo de personajes
## Universo *Lumen: Guardianes Elementales* · DCGAN 64×64

Deep Learning 2026 — Universidad del Valle. Modalidad parejas.

**Integrantes:** Pablo Cabrera (231156) · Luis Palacios (23933)

**Repositorio:** https://github.com/pablouwunya2021/proyecto2-DL
(código, pesos del modelo y galería; permite regenerar la galería desde los pesos y las semillas declaradas).

Entrenamos una GAN que crea **espíritus guardianes bioluminiscentes** nuevos para
un videojuego de ambiente nocturno. El objetivo no es solo "generar algo", sino
producir personajes **nuevos y variados** y demostrarlo con evidencia.

> Descripción completa del universo: [`docs/universo.md`](docs/universo.md).

---

## Estructura del repositorio

```
config.py                  # semillas, hiperparámetros y rutas (configuración base)
src/
  data.py                  # generación procedural del dataset + DataLoader
  models.py                # DCGAN (Generator / Discriminator, con spectral norm opcional)
  losses.py                # BCE no saturante, hinge, WGAN-GP
  train.py                 # loop de entrenamiento + evidencias (ruido fijo, curvas)
  experiments.py           # experimentos de pérdida y estabilización
  features.py              # extractor de características (CNN preentrenada + fallback)
  gallery.py               # selección y generación de la galería final
  nearest_neighbor.py      # prueba de vecinos más cercanos
docs/
  universo.md              # universo visual y origen/licencia de los datos
  hipotesis.md             # hipótesis previas de cada experimento
notebooks/
  proyecto2_lumen_gan.ipynb  # notebook que orquesta todo el pipeline
evidencia/                 # rejillas de ruido fijo, curvas y comparaciones
galeria/                   # 10 personajes finales (PNG) + z_vectors.npz + seleccion.json
checkpoints/               # pesos del generador (G_base.pt, G_hinge.pt, G_spectral.pt)
```

---

## Cómo ejecutar

Requiere Python 3.11+ y las librerías de [`requirements.txt`](requirements.txt).
Usa GPU automáticamente (CUDA o Apple MPS); si no hay, corre en CPU.

```bash
pip install -r requirements.txt

# 1) Generar el dataset (reproducible desde SEED_DATA)
python -m src.data

# 2) Entrenar base + experimentos (base, hinge, spectral) y comparar
python -m src.experiments

# 3) Generar la galería final (10 personajes) desde el modelo base
python -m src.gallery base

# 4) Prueba de vecinos más cercanos
python -m src.nearest_neighbor base
```

O ejecutar el notebook `notebooks/proyecto2_lumen_gan.ipynb` de principio a fin.

### Regenerar la galería desde los pesos
`galeria/z_vectors.npz` guarda el vector **z** de cada personaje y la semilla
`GALLERY_SEED=2026`. Con `G_base.pt` y esos vectores, `python -m src.gallery base`
y `python -m src.nearest_neighbor base` **regeneran exactamente** la galería y la
prueba de vecinos.

---

## Semillas utilizadas

| Semilla         | Valor      | Uso                                             |
|-----------------|------------|-------------------------------------------------|
| `SEED_DATA`     | `20260101` | generación procedural del dataset               |
| `SEED_TRAIN`    | `42`       | inicialización de pesos y orden de los lotes    |
| `SEED_EVAL`     | `7`        | ruido z **fijo** para evaluar/comparar          |
| `GALLERY_SEED`  | `2026`     | muestreo de candidatos para la galería          |

Todos los experimentos comparten `SEED_TRAIN`, épocas (60) y `SEED_EVAL`, y cambian
**una sola cosa a la vez** (ver [`docs/hipotesis.md`](docs/hipotesis.md)).

---

## Datos: origen y licencia

El conjunto de entrenamiento (~5000 imágenes 64×64) es **creación propia**: lo
genera `src/data.py` de forma procedural a partir de `SEED_DATA`, por lo que es
**dominio público (CC0)** y totalmente reproducible. No contiene rostros reales ni
personajes protegidos; los personajes existentes solo inspiraron el estilo general.

---

## Experimentos (resumen)

- **Pérdida:** BCE no saturante (base) vs **hinge** — misma semilla, épocas y ruido.
- **Estabilización:** BatchNorm en D (base) vs **normalización espectral** en D.

Hipótesis previas y criterio de "útil" en [`docs/hipotesis.md`](docs/hipotesis.md);
resultados en `evidencia/` y en la presentación.

---

## Tasa de selección

La galería declara su tasa en `galeria/seleccion.json`: se generaron **200**
muestras y se eligieron **10** descartando las menos novedosas (más cercanas a su
vecino de entrenamiento) y maximizando la diversidad entre las seleccionadas.

---

## Uso de asistentes de IA

Durante el desarrollo usamos un asistente de IA como apoyo puntual para **acelerar
la escritura de código** (plantillas de la DCGAN y utilidades), **depurar** y
**redactar** documentación. El diseño del universo, las decisiones de arquitectura
y de experimentos, la interpretación de las curvas y la selección de la galería son
responsabilidad del equipo, que revisó y verificó todo el código y los resultados.
Ninguna imagen de la galería proviene de un generador externo: todas las produce el
modelo DCGAN entrenado en este repositorio.
