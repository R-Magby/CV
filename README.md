# Portafolio de Proyectos

Gracias por visitar mi repositorio.  
Aquí encontrarás los 4 proyectos más relevantes que he desarrollado.  
Cada carpeta incluye otros trabajos adicionales que muestran mis conocimientos y experiencia.

---

## 1. Simulación N-body con CUDA

- **Lenguajes:** C + CUDA (GPU)
- **Descripción:** Implementé una simulación de N cuerpos para estudiar la dinámica gravitatoria.
- **Métodos utilizados:**
  - Ecuación de Poisson resuelta mediante diferencias finitas.
  - Método *particle-mesh* para obtener la densidad en celdas de tamaño Δx × Δy.
  - Integración temporal con esquema Leapfrog.
- **Optimización:** Ejecución en GPU con CUDA, mejorando el rendimiento para N grandes.
- **Resultado:** Animaciones que muestran la evolución de las partículas formando estructuras y colapsando en un cúmulo central.
  
![Animación](C%20%28CUDA%29/Nbody/animacion_n_cuerpos_.gif)

---

## 2. Predicción de concentración de PM2.5

- **Lenguajes:** Python (Pandas, scikit-learn, matplotlib)
- **Objetivo:** Predecir la calidad del aire en ciudades de Corea del Sur usando datos de contaminantes y variables climáticas.
- **Pasos principales:**
  1. Integración de dos datasets reales (clima y químicos).
  2. Emparejamiento de estaciones por distancia, tipo de ciudad y geografía.
  3. Análisis exploratorio (EDA) y cálculo de correlaciones.
  4. Entrenamiento de modelos Random Forest y SVM con validación cruzada.
- **Resultados:**
  - Accuracy máximo de **63%** en una ciudad.
  - Limitaciones detectadas en categorías de emergencia, atribuibles a calidad de datos y selección de modelo.
- **Aprendizaje:** Proyecto completo desde ETL hasta modelado y evaluación, con potencial de mejora.

![Concentraciones](ciencia%20de%20datos/pm.png)  
![Correlaciones](ciencia%20de%20datos/corr.png)

---

## 3. Software de simulación de ondas

- **Lenguaje:** Python (Tkinter, NumPy, Matplotlib)
- **Objetivo:** Crear una herramienta interactiva y amigable para simular fenómenos ondulatorios.
- **Características:**
  - Ondas estacionarias.
  - Interferencia.
  - Sistemas acoplados.
  - Resorte y otros tipos de ondas.
- **Impacto:** Presentado en charlas universitarias y distribuido gratuitamente a estudiantes para uso académico.
- **Aprendizaje:** Primera experiencia en desarrollo de software con interfaz gráfica, mejorando mis habilidades en simulación física y métodos numéricos.

![Interfaz](TKinter%20%28python%29/img3.png)

---
