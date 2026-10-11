# Hipótesis previas de los experimentos

> Siguiendo la sección 2.1 del enunciado: cada experimento cambia **una sola cosa**
> respecto a la configuración base y mantiene iguales los datos, la semilla
> (`SEED_TRAIN=42`), el número de épocas (60) y el ruido de evaluación
> (`SEED_EVAL=7`). Las hipótesis se escribieron **antes** de ejecutar.

## Configuración base
DCGAN 64×64 · pérdida BCE no saturante · BatchNorm en el discriminador ·
Adam(lr=2e-4, β1=0.5) · batch 128 · 60 épocas.

---

## Experimento A — Función de pérdida: BCE no saturante → *hinge*

**Hipótesis previa:**
Creemos que cambiar la pérdida BCE no saturante por la **pérdida hinge** mejorará
la **estabilidad de los gradientes y la nitidez de las muestras** porque la hinge
deja de penalizar a ejemplos ya bien clasificados (no satura) y produce gradientes
más controlados para el generador. Lo consideraremos útil si las curvas de G y D
se mantienen más estables (menos picos) y las muestras con el ruido fijo se ven al
menos igual de nítidas y variadas que en la base, sin aumentar el colapso de modos.

**Control:** idénticos datos, semilla, épocas y ruido de evaluación; solo cambia la
función de pérdida (`--loss hinge`).

**Resultado y análisis:** _(se completa tras ejecutar — ver `evidencia/expA_*.png` y la presentación)._

---

## Experimento B — Estabilización: BatchNorm en D → Normalización espectral en D

**Hipótesis previa:**
Creemos que añadir **normalización espectral** en el discriminador mejorará la
**estabilidad del entrenamiento y la diversidad** porque acota la constante de
Lipschitz de D, evitando que domine demasiado rápido al generador (una señal de
alarma es D(x)≈1 y D(G(z))≈0 de forma sostenida). Lo consideraremos útil si
D(x) y D(G(z)) se mantienen más cerca de una zona de equilibrio durante más épocas
y la rejilla de ruido fijo muestra más variedad entre filas (menos colapso).

**Control:** idénticos datos, semilla, épocas y ruido de evaluación; solo cambia el
discriminador (BatchNorm → `spectral_norm`, `--spectral-norm`).

**Resultado y análisis:** _(se completa tras ejecutar — ver `evidencia/expB_*.png` y la presentación)._

---

### Nota sobre validez
Un experimento que **no** mejore el resultado también es válido si está bien
ejecutado y analizado (sección 2 del enunciado). El objetivo es entender el efecto
de cada cambio, no forzar una mejora.
