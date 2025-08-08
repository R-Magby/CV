# CV
Muchas gracias por detenerte a leer mi repositorio.
Los codigos presentes los considero los 4 mejores proyectos que he realizado.
Las carpetas contienen más proyectos por si les interesa profundizar en mis conocimientos.
## N-body Simulation
Realicé una simulacion de *N* cuerpos usando el lenguaje C y siendo optimizado con CUDA en la GPU. Este codigo resuelve la ecuacion de Poisson usando diferencias fintias para optener el potencial gravitatorio, tambien suando el metodo mesh para simular un centro gravitatorio en celdas de tamaño $$\Delta x $$ x $$\Delta y $$, Para la evolución se utilizó leapfrog. Se ejecuto con distintos *N* y de manera exitosa en la animacion se muestra como estas orbitan entorno a un cumulo de particulas, generando estructuras, terminando todas juntas. Los ejes son los propios y vs x
![Animacion](C%20%28CUDA%29/Nbody/animacion_n_cuerpos_.gif)
## Prediccion de concnetración de pm2.5
Este proyecto de data science tenia como objetivo unir dos datasets de ciudades de corea del sur y predecir la calidad del aire por medio de machine learning.
Los datasets (url) contienen datos de los quimicos y de variables climaticas, el problema principal es que no todos los datos fueron tomados en las misma estacion, por lo que se tuvo que buscar las estaciones mas parecidas entre si, considerando la distancia, el tipo de ciudad y la geografia del lugar. Luego realizando un EDA  se pudo optener la concnetracion de mateerial particulado y la correlacion entre variables: 
![corre](ciencia%20de%20datos/pm.png)
![corre](ciencia%20de%20datos/corr.png)
Usando modelos de machine learning (Random forest y SVM) se pudo clasificar la calidad del aire usando los criterios recomendados por el Gobierno Chileno, el modelo fue entrenado usando cross validation y obtuvo un accuracy de 63% para una de las ciudades, usando los datos de validación. Segun la matriz de confusion muestra una gran falla en cuanto a la prediccion de las categorias de emergencia. El modelo tiene una presicion muy baja, posiblemente debido a la limpieza y al tipo de modelo, pero puede mejorar, este proyecto muestra mi capacidad de llevar un proyecto de principio a fin.
## Software de simulacion de ondas
Este proyecto universitario tenia como objetivo realizar un ejecutable que sea amigable con el usuario que realice distitnos sistemas fisicos relacionados con fenomenos ondulatorios.
El trabajo se realizó en python, la interfaz con la libreria TKinter y para las simulaciones (calculo y visualizacion) se usaron numpy y matplotlib. Luego de confeccionar el programa, se realizaron charlas a distintos cursos de la misma universidad ofreciendo a estudiantes el producto gratuito para sus estudios. Este proyecto fue enriquesedor para mi debido a que no tenia experiencia en el desarrollo de software y pude enteder lo basico y esencial en poco tiempo, tambien me permitio avanzar en las simualciones de sistemas fisicos y metodos numericos.
![corre](TKinter%20%28python%29/img3.png)
