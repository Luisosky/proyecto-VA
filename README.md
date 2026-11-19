# 🐾 Auditoría Híbrida de Modelos de Visión ante Desafíos de Robustez en Entornos Dinámicos Reales

Proyecto final de **Visión Artificial** (Programa de Ingeniería de Sistemas y Computación) enfocado en la **detección regional y estimación biométrica de tamaño en caninos** en entornos urbanos, comparando en paralelo **Procesamiento Clásico (OpenCV)** vs. **Aprendizaje Profundo (YOLOv8)** bajo condiciones estables y degradadas.

---

## 🎯 Enfoque y Contexto Sociorregional

**Aplicación práctica:** Sistema de monitoreo urbano para censo automatizado, bienestar animal y control vial mediante la detección de caninos en vía pública y la estimación de su porte (**Pequeño**, **Mediano**, **Grande**).

El núcleo del proyecto no se limita a lograr precisión en condiciones ideales, sino a **auditar, comparar y diagnosticar matemáticamente los límites operativos** de ambos enfoques frente a perturbaciones físicas reales (baja iluminación, deslumbramiento, ruido térmico/impulsivo y desenfoque) y proponer **filtros de rescate en preprocesamiento basados en la teoría del píxel**.

---

## 📁 Estructura del Repositorio

```text
proyecto-VA/
├── Proyecto.ipynb                        # Notebook principal estructurado en los 4 módulos de la rúbrica
├── Proyecto Final Visión Artificial.pdf  # Guía oficial, requerimientos y rúbrica de evaluación
├── generar_visor.py                      # Script Python (OpenCV) para generar el visor web interactivo
├── generar_visor.bat                     # Lanzador rápido en Windows (ejecuta/abre el visor HTML)
├── visor_transformaciones.html           # Dashboard web autónomo con las transformaciones paso a paso
├── .gitignore                            # Exclusión de pesos (*.pt), runs/, cachés y entornos virtuales
├── README.md                             # Documentación del proyecto
└── dog.v1i.yolov8/                       # Dataset en formato YOLOv8 (Roboflow, CC BY 4.0)
    ├── data.yaml                         # Configuración de clases y rutas
    ├── train/                            # 2,082 imágenes + etiquetas
    ├── valid/                            # 209 imágenes + etiquetas
    └── test/                             # 103 imágenes + etiquetas
```

---

## 🧩 Módulos Implementados en `Proyecto.ipynb`

El notebook ([Proyecto.ipynb](Proyecto.ipynb)) está diseñado con detección automática de entorno (**Google Colab** o **ejecución local**) y cubre todos los criterios de la rúbrica:

### 1. Módulo 1: Procesamiento Clásico (20%)
* **Fundamentación matemática:** Espacio de color **HSV** (desacople entre matiz y luminancia), histogramas de canales $H$ y $S$, filtrado espacial no lineal (**Mediana**), morfología matemática (**Apertura** y **Cierre**) y descriptores geométricos (**Área de Green/Shoelace**, perímetro, circularidad y relación de aspecto).
* **Estimación biométrica de tamaño:** Clasificación en *Pequeño*, *Mediano* y *Grande* a partir del área segmentada.
* **Métricas obligatorias por frame:** Benchmark sobre `test/images` calculando media ($\mu$), desviación estándar ($\sigma$), percentil 95 ($P_{95}$), **FPS teóricos** y **Tasa de Falsos Positivos (FP/frame)** contrastada mediante $\text{IoU}$ contra el Ground Truth.

### 2. Módulo 2: Aprendizaje Profundo (20%)
* **Transfer Learning con YOLOv8 (`yolov8n.pt`):** Uso de arquitectura *CSPDarknet + C2f*, cuello *PANet* y cabeza *Anchor-Free* con parada temprana (`patience=15`).
* **Curvas obligatorias:** Extracción automática desde `results.csv` para graficar **Box Loss**, **DFL Loss**, **Precision**, **Recall**, **mAP@50** y **mAP@50-95** (entrenamiento vs. validación) con diagnóstico de sobreajuste (*overfitting*).
* **Inferencia en lote:** Medición de latencia por frame y clasificación de tamaño por área de Bounding Box ($w \times h$).

### 3. Módulo 3: Análisis de Robustez y Rigor Métrico (20%)
* **Banco de perturbaciones físicas:** Subexposición nocturna, sobreexposición solar, ruido Gaussiano térmico, ruido impulsivo Sal y Pimienta, desenfoque óptico y entorno crítico mixto.
* **Métricas obligatorias:**
  * **4 Matrices de Confusión Cruzadas** ($2 \times 2$): Clásico vs. YOLOv8 en entorno estable vs. crítico.
  * **Curvas ROC y cálculo de AUC** por integración trapezoidal.
  * **F1-Score bajo condiciones críticas** y tabla comparativa de tiempos por frame.

### 4. Módulo 4: Defensa ante Ataque Adversario y Perturbaciones (40%)
* **Diagnóstico matemático a nivel de píxel:** Explicación formal del colapso en visión clásica (caída de $V$ e indeterminación de $H$) y en redes convolucionales (atenuación de gradientes $\|\nabla I'\| = \alpha \|\nabla I\|$ por debajo del umbral de activación SiLU).
* **Pipeline de rescate en preprocesamiento (`pipeline_defensa_preprocesamiento`):**
  1. **Filtro Bilateral:** Supresión de ruido conservando los gradientes estructurales de bordes.
  2. **Corrección Gamma ($\gamma = 0.65$):** Expansión no lineal de sombras oscuras mediante LUT.
  3. **CLAHE en espacio CIE-Lab:** Ecualización adaptativa limitada al canal de luminancia ($L$) preservando los canales cromáticos ($a, b$).
* **Validación en vivo:** Comparativa visual de 4 etapas (Limpio $\to$ Ataque/Fallo $\to$ Imagen Defendida $\to$ Inferencia Rescatada) y verificación mediante histogramas de intensidad $[0, 255]$.

---

## 📊 Dataset

* **Fuente:** [Roboflow Universe — dog-moyp5 (v1)](https://universe.roboflow.com/vishnu-raj-ufx4d/dog-moyp5/dataset/1) (Licencia **CC BY 4.0**).
* **Total:** 2,394 imágenes redimensionadas a `640x640` en formato YOLOv8:
  * **Train:** 2,082 imágenes y etiquetas.
  * **Valid:** 209 imágenes y etiquetas.
  * **Test:** 103 imágenes y etiquetas.
* **Clase anotada:** `0` (`dog` / `DOG - v2 2025-02-04 2-23pm`), utilizada para la detección regional y contrastada con los descriptores de tamaño geométrico.

---

## 🚀 Cómo Usar el Proyecto

### Opción A: Ejecutar el Notebook en Google Colab (Recomendado para entrenamiento GPU)
1. Sube la carpeta `dog.v1i.yolov8/` a tu Google Drive (`/content/drive/MyDrive/dog.v1i.yolov8`).
2. Abre `Proyecto.ipynb` en **Google Colab** con acelerador **GPU (T4)**.
3. Ejecuta las celdas en orden secuencial. El notebook detecta automáticamente el entorno Colab y monta Drive.

### Opción B: Ejecución Local (Python)
1. Clona el repositorio e instala las dependencias necesarias:
   ```bash
   git clone https://github.com/Luisosky/proyecto-VA.git
   cd proyecto-VA
   pip install ultralytics opencv-python numpy matplotlib seaborn pandas
   ```
2. Abre `Proyecto.ipynb` en Jupyter / VS Code; las rutas se ajustarán automáticamente al directorio local `./dog.v1i.yolov8`.

### Opción C: Visor Didáctico HTML de Transformaciones
Para inspeccionar de manera visual e interactiva cada transformación sin abrir el notebook:
* **En Windows (doble clic):** Ejecuta [`generar_visor.bat`](generar_visor.bat) para generar/abrir [`visor_transformaciones.html`](visor_transformaciones.html) en tu navegador.
* **Desde terminal con otra imagen:**
  ```bash
  python generar_visor.py "ruta/a/tu_imagen.jpg"
  ```
* **Directamente:** Abre [`visor_transformaciones.html`](visor_transformaciones.html) en cualquier navegador (es 100% autónomo y funciona sin conexión).
