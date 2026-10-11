# Universo visual — *Lumen: Guardianes Elementales*

## Concepto
**Lumen** es un bestiario de pequeños **espíritus guardianes bioluminiscentes**
para un videojuego de exploración de ambiente nocturno. Cada guardián es una
criatura compacta y simétrica que flota en la oscuridad, con un **núcleo de luz**
en el pecho y una paleta de color ligada a su **elemento**.

## Estilo y ambiente
- **Fondo:** siempre oscuro, con una tenue viñeta del color elemental del guardián.
- **Forma:** silueta vertical, centrada y **simétrica**; lectura de "avatar/token".
- **Luz:** un núcleo brillante y un halo suave (bioluminiscencia) definen al personaje.
- **Resolución:** 64×64 px (recomendación del enunciado; coherente y económica).

## Rasgos que esperamos ver en la galería
- Siluetas reconocibles de uno de cuatro **arquetipos**: `blob`, `humanoide`,
  `bestia`, `fulgor` (llama/gota).
- **Ojos** simétricos (1–3) y, con frecuencia, **cuernos o antenas**.
- Un **núcleo** luminoso central y un halo del mismo tono.
- Pertenencia visual a uno de seis **elementos**: `brasa` (fuego), `marea` (agua),
  `musgo` (tierra), `céfiro` (aire), `vacío` (sombra), `solar` (luz).

## Vara de evaluación
La galería final se juzga contra esta descripción: un personaje "bueno" se lee como
un guardián Lumen —simétrico, con núcleo brillante, paleta elemental coherente— y a
la vez es **nuevo** (distinto de las imágenes de entrenamiento) y **distinto** de los
otros nueve.

## Origen y licencia de los datos
El conjunto de entrenamiento **no** usa personajes existentes ni rostros reales: se
genera por completo con nuestro script procedural (`src/data.py`) a partir de una
semilla (`SEED_DATA=20260101`). Al ser creación propia, el material es de **dominio
público (CC0)** y totalmente reproducible. Los personajes existentes solo sirvieron
como inspiración de estilo general (criaturas bioluminiscentes de videojuegos), no
se copiaron ni aparecen en los datos.
