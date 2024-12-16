Esta carpeta tiene algunos proyectos que he realizado en el curso "Optimización con GPU", donde aprendí a desarrollar códigos usando CUDA en C.

Los códigos y GIF que subí son simulaciones de sistemas físicos como:

- **Simulación de N-cuerpos:** Paralelicé el cálculo de la evolución de cada partícula utilizando el método *Leapfrog* y la ecuación de Poisson con diferencias finitas.

- **Ecuación de onda:** Paralelicé el cálculo de cada punto espacial de la grilla de \(\phi\) usando diferencias finitas. También implementé *streams* para calcular dos ondas por separado y, usando la linealidad de la ecuación, sumé ambas soluciones.

- **Simulación del colapso gravitacional de un campo escalar:** Utilicé el formalismo ADM para paralelizar varias ecuaciones para su evolución. El GIF muestra un colapso con una amplitud inicial de 0.01. Cuando la onda deja de avanzar en el tiempo, indica que la función lapso de la métrica es cero, es decir, no hay paso de tiempo propio en el agujero negro.

Estos proyectos demuestran mi capacidad para optimizar simulaciones físicas y paralelizar algoritmos complejos utilizando CUDA.
