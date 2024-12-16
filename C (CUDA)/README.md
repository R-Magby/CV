Esta carpeta tiene algunos proyectos que he realizado en el curso "optimizacion con gpu", donde aprendí desarrollar codigos usando cuda en c.
Los codigos y gif que subí son simulaciones de sistemas fisicos como:\\
-simulacion de n cuerpos donde paralelicé el calculo de la evolucion de cada particula, usé leapfrog y la ecuacion de poisson con diferencias finitas.\\
-la ecuacion de onda, paralelicé el calculo de cada punto espacial de la grilla de $\phi$ usando diferencias finitas, tambien implemente stream para
  calcular 2 ondas por separados y usando la linealidad de la ecuacion, sume ambas soluciones.\\
-una simulacion del colapso gravitacional de un campo escalar usando el formalismo ADM, donde paralelicé varias ecuaciones para su evolucion, el caso de los gif es un colapso
  con amplitud inicial de 0.01, cuando la onda deja de avanzar en el tiempo indica que la funcion lapso de la metrica es cero, es decir no hay paso de tiempo
  propio de un agujero negro.
