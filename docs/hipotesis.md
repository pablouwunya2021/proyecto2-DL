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

**Resultado y análisis:**
Al final de las 60 épocas la base (BCE) terminó en `D(x)=0.86`, `D(G(z))=0.14`,
`loss_G≈3.0`; la variante hinge en `D(x)=0.80`, `D(G(z))=0.20`, `loss_G≈2.2`.
La pérdida de G con hinge es más baja y **más plana** (menos oscilación época a
época, ver `evidencia/expA_perdida_Lg.png`) y `D(G(z))` se mantiene algo más alto
y estable (`expA_perdida_Dfake.png`), es decir, el discriminador domina un poco
menos. La nitidez y variedad de la rejilla de ruido fijo son comparables entre
ambas. **Conclusión:** la hipótesis se confirma parcialmente: hinge sí dio
gradientes/curvas más estables, pero a 60 épocas **no** produjo una mejora de
calidad visual claramente perceptible. Útil como estabilizador, no como salto de
calidad en este universo.

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

**Resultado y análisis:**
Efecto claro y en la dirección esperada. La base (BatchNorm) sostiene
`D(x)≈0.80 / D(G(z))≈0.20` y tiende a separarse más al final
(`D(x)=0.86` en la época 60). Con normalización espectral, `D(x)` y `D(G(z))`
**convergen hacia el equilibrio**: terminan en `0.56 / 0.44`, mucho más cerca del
ideal teórico de 0.5, y `loss_D` sube a ≈1.2 (ver `evidencia/expB_estab_Dreal.png`
y `expB_estab_Ld.png`). Esto es justo lo que predice la teoría: acotar la constante
de Lipschitz de D evita que domine al generador. **Conclusión:** la hipótesis se
confirma: la normalización espectral mantiene el entrenamiento en una zona de
equilibrio más sana durante más épocas. (No la usamos para la galería final solo
porque el modelo base ya daba muestras nítidas; como estabilizador es la opción más
sólida si se escala a más épocas o mayor resolución.)

---

### Nota sobre validez
Un experimento que **no** mejore el resultado también es válido si está bien
ejecutado y analizado (sección 2 del enunciado). El objetivo es entender el efecto
de cada cambio, no forzar una mejora.
